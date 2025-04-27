#app create by Salvatore Naro
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import threading
from tkinter import messagebox
import customtkinter
from PIL import Image, ImageSequence
import tkinter
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.rete = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 4)
        )
    def calcoli(self,y:np.array)->np.array:
        device = y.device
        array = np.array([[1,2,3,4],[5,6,7,8]],device=device)
        array2= np.array([[9,10,11,12],[13,14,15,16]],device=device)
        y_arraytot = np.prod(array) + np.prod(array2)
        arrayconcatenate = np.concatenate((array,array2),axis=0)
        print(f"Array concatenate: {arrayconcatenate}")
        print(f"array:{y_arraytot}")
        return self.rete(y_arraytot)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        tensore = torch.tensor([[1., 2., 3., 4., 5.],
                                [6., 7., 8., 9., 10.]], device=device)
        tensorey = torch.tensor([[11., 12., 13., 14., 15.],
                                 [16., 17., 18., 19., 20.]], device=device)
        f = tensorey * tensore
        x_strided = torch.as_strided(tensore, (2, 4), (5, 1)) 
        k = torch.as_strided(f, (2, 4), (5, 1)) 

        tensore[:, -1] *= 2 
        ones = torch.ones([2, 3, 4], device=device)
        zeros = torch.zeros([2, 3, 4], device=device)
        randx = torch.rand([2, 3, 4], device=device)
        randy = torch.rand([4, 2, 3, 1], device=device)
        randtot = torch.sin(randx) * torch.sin(randy)

        
        print("randx shape:", randx.shape)
        print("randx reshaped:", randx.view(1, 24))
        print("randy stride:", randy.stride())
        print("randtot stride:", randtot.stride())
        print("zeros shape:", zeros.shape)
        print("ones shape:", ones.shape)
        print("ones unsqueezed * 3:", torch.unsqueeze(ones, dim=1) * 3)
        print("tensore reshaped:", tensore.reshape([1, 10]))
        print("k (strided f):", k)
        print("k stride:", k.stride())
        print("x_strided:", x_strided)

        
        return self.rete(x_strided)


def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = Network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            data = data[:, :4] 
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)  
            dummy_target = torch.randn(output.shape, device=device)  
            loss = criterion(output, dummy_target)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

customtkinter.set_appearance_mode('dark')
window = customtkinter.CTk()
window.geometry("600x650")
#window.resizable(False,False)
window.columnconfigure(0,weight=1)
window.config(bg="black")


gif_path = "C:\\Users\\salva\\Downloads\\sakrim.gif"
gif = Image.open(gif_path)

frames = []
for frame in ImageSequence.Iterator(gif):
    frames.append(customtkinter.CTkImage(light_image=frame.copy(),size=(400, 400)))
    
label = customtkinter.CTkLabel(window, text="",bg_color="black",fg_color="black")
label.pack()
frame = customtkinter.CTkFrame(master=window,bg_color="black",fg_color="black",width=400,height=400)
frame.pack(padx=40,pady=10)
nome = customtkinter.CTkEntry(master=frame,placeholder_text="Nome",width=150,fg_color="Black",text_color="white")
nome.pack(side="left",padx=45,pady=2)
cognome = customtkinter.CTkEntry(master=frame,placeholder_text="Cognome",width=150,fg_color="Black",text_color="white")
cognome.pack(side="right",padx=45,pady=2)
frame2 = customtkinter.CTkFrame(master=window,bg_color="black",fg_color="black",width=350,height=350)
frame2.pack(padx=40,pady=15)
lobel = customtkinter.CTkLabel(master=frame2,text="Inizia l'addestramento",font=("Arial",30))
lobel.pack(padx=45,pady=10)
def app():
    numeri = "1234567890!£$%€%&/()=?^°|[]{};:_-.,<>@#"
    nomes = nome.get()
    cognomes = cognome.get()
    if any(char in numeri for char in nome.get()) or any(char in numeri for char in cognome.get()):
        messagebox.showerror(title="Errore",message="Non puoi inserire numeri o simboli speciali")
    if (nomes=="") or (cognomes==""):
        messagebox.showerror(title="Errore",message="Non puoi lasciare vuoti i campi")
    else:
        threading.Thread(target=train_model).start()
def apps(event):
    numeri = "1234567890!£$%€%&/()=?^°|[]{};:_-.,<>@#"
    nomes = nome.get()
    cognomes = cognome.get()
    if any(char in numeri for char in nomes)or any(char in numeri for char in cognomes):
        messagebox.showerror(title="Errore",message="Non puoi inserire numeri o simboli speciali")
    if (nomes=="") or (cognomes==""):
        messagebox.showerror(title="Errore",message="Non puoi lasciare vuoti i campi")
    else:
        threading.Thread(target=train_model).start()
def writes():
    messagebox.showinfo(title="Informazioni",message="L'app è ancora in fase di sperimentazione è simula una rete neurale abbastanza complessa!")
button = customtkinter.CTkButton(master=frame2,text="Start",fg_color="red",command=app)
button.pack(side="left",padx=45,pady=5)
write = customtkinter.CTkButton(master=frame2,text="WRITE",fg_color="black",text_color="white",command=writes)
write.pack(side="right",padx=49,pady=5)
window.bind("<Return>",apps)

def animate(frame_num=0):
    frame_num = (frame_num + 1) % len(frames)
    label.configure(image=frames[frame_num])
    window.after(10, animate, frame_num)  

animate()  
window.mainloop()
if __name__ == "__main__":
        set_seed(42)
        model = Network()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = torch.randn(2, 4, device=device)
        input_numpy = np.random.randint(0, 10, size=(2, 4))  
        input_numpy_tensor = torch.from_numpy(input_numpy).float().to(device)  
        output = model(input_tensor)
        output2 = model(input_numpy_tensor)
        print(f"Output shape (input_tensor): {output.shape}")
        print(f"Output (input_tensor): {output}")
        print(f"Output shape (input_numpy): {output2.shape}")
        print(f"Output (input_numpy): {output2}")
        train_model()