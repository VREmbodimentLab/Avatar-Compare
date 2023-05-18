import socket
import numpy as np
import torch
import torch.nn as nn
import time
from models.network import AvatarPoser as net
HOST = '127.0.0.1'  
PORT = 5000        

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print('Server Started')
client_socket, addr = server_socket.accept()
print('Client connected:', addr)

# Network model 정의
pretrained_model = torch.load('./model_zoo/avatarposer.pth')
model = net()
model.load_state_dict(pretrained_model)
model.eval()

while True:
    # Data reception from client
    msg = client_socket.recv(1024)
    message = np.fromstring(msg, dtype=np.float32, sep=',')
    # message = np.reshape(message, (window_size, -1))
    start = time.time()

    input = torch.from_numpy(message).float()
    input = torch.unsqueeze(input, 0)
    print("the input value is :", input)

    # Input data modify
    


    output = model(input)


    # Data sending to client
    response_data = '서버에서 보내는 응답 데이터입니다.'
    client_socket.sendall(response_data.encode())
    
    client_socket.close()

