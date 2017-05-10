# coding=utf8
import socket


#   Адресс + порт на который отправляем
host_address = 'localhost'
port = 4010

#   Сообщение которое отправляем
MESSAGE = 'I love Katty! :)'

#   Количество раз, сколько раз последовательно мы отправляем сообщение
NUMBER=25

for i in range(100):
    try:
        buffer_size=100
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host_address, port))
        s.send(MESSAGE)
        data = s.recv(buffer_size)
        s.close()

        print "received data:", data
    except:
        print 'Error!!!!!!!!'
        pass