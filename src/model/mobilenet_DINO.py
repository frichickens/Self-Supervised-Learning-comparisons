import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mobilenet import MobileNetV1

from utils.dino_utils import DINOHead


class MobileNet_DINO(nn.Module):
    """DINO with MobileNet backbone"""
    def __init__(self, c_in=3, out_dim=65536, use_bn_in_head=False, 
                norm_last_layer=True, hidden_dim=2048, bottleneck_dim=256, 
                nlayers=3, momentum_teacher=0.996):
        super().__init__()
        
        # Student network - use existing MobileNetV1 without classifier
        student_mobilenet = MobileNetV1(c_in=c_in, c_out=1024)
        self.student_backbone = nn.Sequential(*list(student_mobilenet.children())[:-1], nn.Flatten())  # Remove classifier
        self.student_backbone.out_dim = 1024
        
        self.student_head = DINOHead(
            self.student_backbone.out_dim, 
            out_dim, 
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Teacher network - use existing MobileNetV1 without classifier
        teacher_mobilenet = MobileNetV1(c_in=c_in, c_out=1024)
        self.teacher_backbone = nn.Sequential(*list(teacher_mobilenet.children())[:-1], nn.Flatten())  # Remove classifier
        self.teacher_backbone.out_dim = 1024
        
        self.teacher_head = DINOHead(
            self.teacher_backbone.out_dim,
            out_dim,
            use_bn=use_bn_in_head,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Initialize teacher with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Freeze teacher
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
            
        self.momentum_teacher = momentum_teacher

    @torch.no_grad()
    def update_teacher(self, momentum=None):
        """EMA update of the teacher network"""
        if momentum is None:
            momentum = self.momentum_teacher
        for param_s, param_t in zip(self.student_backbone.parameters(), 
                                    self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
        for param_s, param_t in zip(self.student_head.parameters(), 
                                    self.teacher_head.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)

    def forward(self, x_student, x_teacher=None, return_backbone=False):
        """
        Args:
            x_student: student input (multi-crop views)
            x_teacher: teacher input (global views only)
            return_backbone: if True, return backbone features only
        """
        if return_backbone:
            return self.student_backbone(x_student)
            
        # Student forward
        student_features = self.student_backbone(x_student)
        student_output = self.student_head(student_features)
        
        if x_teacher is None:
            return student_output
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_features = self.teacher_backbone(x_teacher)
            teacher_output = self.teacher_head(teacher_features)
        
        return student_output, teacher_output

