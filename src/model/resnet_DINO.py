import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import ResNet18, ResNet34


class DINOHead(nn.Module):
    """Projection head for DINO"""
    def __init__(self, in_dim, out_dim=65536, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class ResNetBackbone(nn.Module):
    """Wrapper for ResNet backbone without classification head"""
    def __init__(self, resnet_model):
        super().__init__()
        # Get all children modules as a list
        modules = list(resnet_model.children())
        
        # Remove the last FC layer - typically it's the last module
        # ResNet structure usually ends with: ..., avgpool, fc
        # We want everything except fc
        self.features = nn.Sequential(*modules[:-1])
        
        # Add adaptive pooling and flatten
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class ResNet18_DINO(nn.Module):
    """DINO with ResNet18 backbone"""
    def __init__(self, out_dim=65536, use_bn_in_head=False, 
                 norm_last_layer=True, hidden_dim=2048, bottleneck_dim=256, 
                 nlayers=3, momentum_teacher=0.996):
        super().__init__()
        
        # Create student network (ResNet18 backbone)
        student_model = ResNet18(c_out=10)
        self.student_backbone = ResNetBackbone(student_model)
        
        # ResNet18 outputs 512-dim features
        backbone_dim = 512
        
        # Student projection head
        self.student_head = DINOHead(
            backbone_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Create teacher network
        teacher_model = ResNet18(c_out=10)
        self.teacher_backbone = ResNetBackbone(teacher_model)
        
        # Teacher projection head
        self.teacher_head = DINOHead(
            backbone_dim,
            out_dim,
            use_bn=use_bn_in_head,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Initialize teacher with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Freeze teacher network
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
        
        # Update teacher backbone
        for param_s, param_t in zip(self.student_backbone.parameters(), 
                                     self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
        
        # Update teacher head
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


class ResNet34_DINO(nn.Module):
    """DINO with ResNet34 backbone"""
    def __init__(self, out_dim=65536, use_bn_in_head=False, 
                 norm_last_layer=True, hidden_dim=2048, bottleneck_dim=256, 
                 nlayers=3, momentum_teacher=0.996):
        super().__init__()
        
        # Create student network (ResNet34 backbone)
        student_model = ResNet34(c_out=10)
        self.student_backbone = ResNetBackbone(student_model)
        
        # ResNet34 outputs 512-dim features
        backbone_dim = 512
        
        # Student projection head
        self.student_head = DINOHead(
            backbone_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Create teacher network
        teacher_model = ResNet34(c_out=10)
        self.teacher_backbone = ResNetBackbone(teacher_model)
        
        # Teacher projection head
        self.teacher_head = DINOHead(
            backbone_dim,
            out_dim,
            use_bn=use_bn_in_head,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Initialize teacher with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Freeze teacher network
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
        
        # Update teacher backbone
        for param_s, param_t in zip(self.student_backbone.parameters(), 
                                     self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
        
        # Update teacher head
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


class DINOLoss(nn.Module):
    """DINO loss with centering and sharpening"""
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Temperature schedule for teacher
        self.teacher_temp_schedule = torch.cat([
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ])

    def forward(self, student_output, teacher_output, epoch):
        """Cross-entropy between softmax outputs of the teacher and student networks."""
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip same crop
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)