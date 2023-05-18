import socket
import numpy as np
import torch
import torch.nn as nn

HOST = '127.0.0.1'  
PORT = 5000        

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print('서버가 시작되었습니다. 클라이언트 연결을 기다립니다.')
client_socket, addr = server_socket.accept()
print('클라이언트가 연결되었습니다:', addr)

while True:
    # 데이터 수신
    data = client_socket.recv(1024)
    if not data:
        break

    # 수신된 데이터 처리
    received_data = data.decode()
    print('받은 데이터:', received_data)

# 클라이언트 소켓과 연결 종료
client_socket.close()