import torch.nn as nn
import torch
from torchvision import transforms, datasets
import sys
import pygame
import Model

def guess():
    
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((224,224)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    img_dir = r"Img"
    
    img_data = datasets.ImageFolder(img_dir,transform = transform)
    
    img_loader = torch.utils.data.DataLoader(img_data)
      
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load('checkpoint44.pt', weights_only=True)
    
    model = nn.DataParallel(Model.MobileNetV1(ch_in = 3, n_classes = 10), device_ids = [0]).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    class_labels = ['AIRPLANE','APPLE','ARM','BANANA',
                    'BIRD','CAR','CAT','DOG','FISH','FOOT']
    
    model.eval()
    with torch.no_grad():
        for image, _ in img_loader:
            image = image.to(device)
            outputs = model(image)
            print((nn.functional.softmax(outputs, dim = 1)).cpu().numpy()*100)
            _, predicted = torch.max(outputs.data, 1)
            del image, outputs
            return(class_labels[predicted])

pygame.init()

pygame_icon = pygame.image.load('Icon.png')
pygame.display.set_icon(pygame_icon)

H = 400
W = 600

screen = pygame.display.set_mode([W,H])
screen.fill('black')

pygame.display.set_caption('Paint!')

drawing = False
last_pos = None

mouse_position = (0,0)

font = pygame.font.Font(r'C:\Windows\Fonts\ARLRDBD.ttf', 20)

guess_text = font.render('I THINK IT IS A', True, (226, 241, 231),'black')
guess_rect = guess_text.get_rect()
guess_rect.center = (500,150)

guessing = False

answer_text = None
answer_width = None
answer_height = None
answer_x_pos = None
answer_y_pos = None

    
while True:
    
    active_size = 5
    active_color = 'white'           
    
    #Guess bar
    pygame.draw.rect(screen, (36, 54, 66),[400,0,400,400])
    screen.blit(guess_text, guess_rect)
    
    if guessing:
        screen.blit(answer_text, (answer_x_pos, answer_y_pos))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            if (drawing):
                mouse_position = pygame.mouse.get_pos()
                if last_pos is not None:
                    if mouse_position[0]<400:
                        pygame.draw.line(screen, 'white', last_pos, mouse_position, active_size)
                    
                last_pos = mouse_position
                
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_position = (0, 0)
            drawing = False
            last_pos = None
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            
            if event.button == 1:
                drawing = True
                guessing = False
                
            elif event.button == 3:
                guessing = False
                screen.fill('black')
                
            elif event.button == 2:
                rect = pygame.Rect(0, 0, 400, 400)
                crop = screen.subsurface(rect)
                pygame.image.save(crop, r"Img/This folder/painting.png") 
                print("Processing")
                font2 = pygame.font.Font(r'C:\Windows\Fonts\ARLRDBD.ttf', 35)
                answer_text = font2.render('{}'.format(guess()), True, (98, 149, 132),'black')
                answer_width = answer_text.get_width()
                answer_height = answer_text.get_height()
                answer_x_pos = 500 - answer_width // 2
                answer_y_pos = 250 - answer_height // 2
                print("==================")
                guessing = True
    
    pygame.display.update()

pygame.quit()