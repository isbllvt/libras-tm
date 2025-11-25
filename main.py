import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

class ReconhecedorLIBRAS:
    def __init__(self, model_path="model/keras_model.h5", labels_path="model/labels.txt"):
        """
        Inicializa o reconhecedor de LIBRAS
        """
        try:
            # Carrega o modelo treinado
            self.model = load_model(model_path, compile=False)
            
            # Carrega lista de labels
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
            
            print(f"Modelo carregado com sucesso! Labels: {self.labels}")
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            exit()
        
        # Configurações
        self.confidence_threshold = 0.7  # Limite de confiança
        self.last_prediction = ""
        self.prediction_time = time.time()
        
    def preprocessar_imagem(self, frame):
        """
        Pré-processa a imagem para o modelo
        """
        # Redimensiona para 224x224 (tamanho esperado pelo modelo)
        img = cv2.resize(frame, (224, 224))
        
        # Converte BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normaliza os pixels para [0, 1]
        img = np.array(img) / 255.0
        
        # Adiciona dimensão do batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def prever_gesto(self, frame):
        """
        Faz a predição do gesto na imagem
        """
        img_processada = self.preprocessar_imagem(frame)
        
        # Predição
        prediction = self.model.predict(img_processada, verbose=0)
        confidence = np.max(prediction)
        index = np.argmax(prediction)
        
        return self.labels[index], confidence, index
    
    def executar(self):
        """
        Executa o loop principal de captura e reconhecimento
        """
        # Inicia webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Erro: Não foi possível acessar a webcam")
            return
        
        print("\n=== Reconhecimento de LIBRAS ===")
        print("Instruções:")
        print("- Pressione 'q' para sair")
        print("- Pressione 's' para salvar uma imagem")
        print("- Certifique-se de ter boa iluminação")
        print("- Mantenha a mão no centro da tela")
        
        frame_count = 0
        saved_count = 0
        fps_time = time.time()  # Tempo separado para cálculo de FPS
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Espelha o frame para efeito espelho (mais natural)
            frame = cv2.flip(frame, 1)
            
            # Faz uma cópia para desenhar
            display_frame = frame.copy()
            
            # Desenha área de interesse
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            roi_size = 300
            
            # Define região de interesse (ROI)
            x1 = center_x - roi_size // 2
            y1 = center_y - roi_size // 2
            x2 = center_x + roi_size // 2
            y2 = center_y + roi_size // 2
            
            # Desenha retângulo da ROI
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Area da Mao", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Recorta a ROI para processamento
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Faz a predição apenas na ROI
                gesto, confidence, index = self.prever_gesto(roi)
                
                # Atualiza predição apenas se a confiança for alta
                if confidence > self.confidence_threshold:
                    self.last_prediction = gesto
                
                # Cor baseada na confiança
                color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 0, 255)
                
                # Exibe informações na tela
                cv2.putText(display_frame, 
                           f"Gesto: {self.last_prediction}", 
                           (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
                
                cv2.putText(display_frame, 
                           f"Confianca: {confidence:.2f}", 
                           (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
                
                # Barra de confiança
                bar_width = 200
                confidence_width = int(confidence * bar_width)
                cv2.rectangle(display_frame, (10, 100), (10 + bar_width, 120), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (10, 100), (10 + confidence_width, 120), color, -1)
                
            # Exibe frame rate (CORRIGIDO)
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                time_diff = current_time - fps_time
                
                if time_diff > 0:
                    fps = 30 / time_diff
                else:
                    fps = 0
                    
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, height-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                fps_time = current_time  # Atualiza para próximo cálculo
            
            cv2.imshow("Reconhecimento LIBRAS - Pressione 'q' para sair", display_frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salva imagem atual
                filename = f"captura_{saved_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Imagem salva como {filename}")
                saved_count += 1
        
        # Libera recursos
        cap.release()
        cv2.destroyAllWindows()
        print("Programa encerrado.")
def main():
    """
    Função principal
    """
    # Verifica se os arquivos do modelo existem
    import os
    if not os.path.exists("model/keras_model.h5") or not os.path.exists("model/labels.txt"):
        print("Erro: Arquivos do modelo não encontrados!")
        print("Certifique-se de que a pasta 'modelo' contém:")
        print("- keras_model.h5")
        print("- labels.txt")
        return
    
    # Inicializa e executa o reconhecedor
    reconhecedor = ReconhecedorLIBRAS()
    reconhecedor.executar()

if __name__ == "__main__":
    main()