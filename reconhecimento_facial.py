import cv2
import face_recognition
import os
import numpy as np

# Função para capturar imagens e cadastrar um rosto
def registrar_rosto(nome):
    video_capture = cv2.VideoCapture(0)
    print(f"Cadastrando o rosto de {nome}. Aperte 'q' para capturar a imagem.")

    while True:
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)

        # Captura o frame quando pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rosto = frame
            break

    # Converte a imagem para RGB
    rosto_rgb = cv2.cvtColor(rosto, cv2.COLOR_BGR2RGB)

    # Processa e pega a codificação do rosto
    face_encoding = face_recognition.face_encodings(rosto_rgb)[0]

    # Salva a codificação em um arquivo .npy
    np.save(f'rostos/{nome}.npy', face_encoding)
    print(f"Rosto de {nome} cadastrado com sucesso!")

    video_capture.release()
    cv2.destroyAllWindows()

# Função para reconhecer rostos em tempo real
def reconhecer_rosto():
    rostos_cadastrados = []
    nomes = []

    # Carrega todos os rostos cadastrados na pasta
    for arquivo in os.listdir('rostos'):
        if arquivo.endswith('.npy'):
            face_encoding = np.load(f'rostos/{arquivo}')
            rostos_cadastrados.append(face_encoding)
            nomes.append(arquivo.replace('.npy', ''))

    video_capture = cv2.VideoCapture(0)
    print("Reconhecendo rostos. Aperte 'q' para sair.")

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detecta os rostos no frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(rostos_cadastrados, face_encoding)
            nome = "Desconhecido"

            face_distances = face_recognition.face_distance(rostos_cadastrados, face_encoding)
            melhor_match = np.argmin(face_distances)

            if matches[melhor_match]:
                nome = nomes[melhor_match]

            print(f"Rosto identificado: {nome}")

        # Exibe o vídeo com a detecção
        cv2.imshow('Video', frame)

        # Para o vídeo ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Menu
if __name__ == "__main__":
    if not os.path.exists('rostos'):
        os.makedirs('rostos')

    print("1. Cadastrar novo rosto")
    print("2. Reconhecer rostos")

    escolha = input("Escolha uma opção: ")

    if escolha == '1':
        nome = input("Digite o nome da pessoa: ")
        registrar_rosto(nome)
    elif escolha == '2':
        reconhecer_rosto()
    else:
        print("Opção inválida!")
