import random
from unicodedata import name
import torch
import pandas as pd
import re
import csv
import numpy as np
from utils import load_model
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time 

tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case = True)
model = BertForSequenceClassification.from_pretrained(
                'dccuchile/bert-base-spanish-wwm-uncased', 
                num_labels = 2, 
                output_attentions = False, 
                output_hidden_states = False, 
        )    

class ClaimClassifier:
    def __init__(self) -> str:
        
        self.tipo_mercado = load_model(model, tokenizer, 'models/mercado_ingreso_bancos_o_otros', 'cpu')
        self.tipo_mercado_seguros_valores = load_model(model, tokenizer, 'models/mercado_ingreso_seguros_o_valores', 'cpu')
        
        self.bancos_tipo_entidad = load_model(model, tokenizer, 'models/bancos_tipo_entidad', 'cpu')
        self.bancos_tipo_producto = load_model(model, tokenizer, 'models/bancos_tipo_producto', 'cpu')
        self.bancos_tipo_materia = load_model(model, tokenizer, 'models/bancos_tipo_materia', 'cpu')
        self.bancos_nombre_entidad = load_model(model, tokenizer, 'models/bancos_nombre_entidad', 'cpu')
        
        self.tipo_mercado.eval()
        self.tipo_mercado_seguros_valores.eval()
        self.bancos_tipo_entidad.eval()
        self.bancos_tipo_producto.eval()
        self.bancos_tipo_materia.eval()
        self.bancos_nombre_entidad.eval()

        self.seguros_valores_tipo_entidad = load_model(model, tokenizer, 'models/seguros_valores_tipo_entidad', 'cpu')
        self.seguros_valores_tipo_producto = load_model(model, tokenizer, 'models/seguros_valores_tipo_producto', 'cpu')
        self.seguros_valores_tipo_materia = load_model(model, tokenizer, 'models/seguros_valores_tipo_materia', 'cpu')
        self.seguros_valores_nombre_entidad = load_model(model, tokenizer, 'models/seguros_valores_nombre_entidad', 'cpu')
     
        self.seguros_valores_tipo_entidad.eval()
        self.seguros_valores_tipo_producto.eval()
        self.seguros_valores_tipo_materia.eval()
        self.seguros_valores_nombre_entidad.eval()

    def predict(self, sentences: list):
        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            
    
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
            
        # Predicción por oración, una lista de diccionarios

        # Set the batch size.  
        batch_size = 1  

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        
       
        predictions , true_labels = [], []
        table = []

        device = 'cpu'

        predictions = []

        for i, batch in enumerate(prediction_dataloader):
            sentence_data = []
            
           
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
        
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                
                outputs_tipo_mercado = self.tipo_mercado(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

                outputs_tipo_mercado_seguros_valores = self.tipo_mercado_seguros_valores(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

                outputs_bancos_tipo_producto = self.bancos_tipo_producto(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
                
                outputs_bancos_tipo_entidad = self.bancos_tipo_entidad(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
                
                outputs_bancos_tipo_materia = self.bancos_tipo_materia(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
                
                outputs_bancos_nombre_entidad = self.bancos_nombre_entidad(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

                
                outputs_seguros_valores_tipo_producto = self.seguros_valores_tipo_producto(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
                
                outputs_seguros_valores_tipo_entidad = self.seguros_valores_tipo_entidad(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
                
                outputs_seguros_valores_tipo_materia = self.seguros_valores_tipo_materia(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
                
                outputs_seguros_valores_nombre_entidad = self.seguros_valores_nombre_entidad(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

                

            logits_tipo_mercado = outputs_tipo_mercado[0]
            logits_tipo_mercado_seguros_valores = outputs_tipo_mercado_seguros_valores[0]

            logits_bancos_tipo_entidad = outputs_bancos_tipo_entidad[0]
            logits_bancos_tipo_producto = outputs_bancos_tipo_producto[0]
            logits_bancos_tipo_materia = outputs_bancos_tipo_materia[0]
            logits_bancos_nombre_entidad = outputs_bancos_nombre_entidad[0]
            
            logits_seguros_valores_tipo_entidad = outputs_seguros_valores_tipo_entidad[0]
            logits_seguros_valores_tipo_producto = outputs_seguros_valores_tipo_producto[0]
            logits_seguros_valores_tipo_materia = outputs_seguros_valores_tipo_materia[0]
            logits_seguros_valores_nombre_entidad = outputs_seguros_valores_nombre_entidad[0]


            # Move logits and labels to CPU
            logits_tipo_mercado = logits_tipo_mercado.detach().cpu().numpy()
            pred_tipo_mercado = np.argmax(logits_tipo_mercado, axis=1).flatten()

            # Move logits and labels to CPU
            logits_tipo_mercado_seguros_valores = logits_tipo_mercado_seguros_valores.detach().cpu().numpy()
            pred_tipo_mercado_seguros_valores = np.argmax(logits_tipo_mercado_seguros_valores, axis=1).flatten()


            # Move logits and labels to CPU
            logits_bancos_tipo_entidad = logits_bancos_tipo_entidad.detach().cpu().numpy()
            pred_bancos_tipo_entidad = np.argmax(logits_bancos_tipo_entidad, axis=1).flatten()


            # Move logits and labels to CPU
            logits_bancos_tipo_producto = logits_bancos_tipo_producto.detach().cpu().numpy()
            pred_bancos_tipo_producto = np.argmax(logits_bancos_tipo_producto, axis=1).flatten()

            # Move logits and labels to CPU
            logits_bancos_tipo_materia = logits_bancos_tipo_materia.detach().cpu().numpy()
            pred_bancos_tipo_materia = np.argmax(logits_bancos_tipo_materia, axis=1).flatten()

            # Move logits and labels to CPU
            logits_bancos_nombre_entidad = logits_bancos_nombre_entidad.detach().cpu().numpy()
            pred_bancos_nombre_entidad = np.argmax(logits_bancos_nombre_entidad, axis=1).flatten()


            # Move logits and labels to CPU
            logits_seguros_valores_tipo_entidad = logits_seguros_valores_tipo_entidad.detach().cpu().numpy()
            pred_seguros_valores_tipo_entidad = np.argmax(logits_seguros_valores_tipo_entidad, axis=1).flatten()


            # Move logits and labels to CPU
            logits_seguros_valores_tipo_producto = logits_seguros_valores_tipo_producto.detach().cpu().numpy()
            pred_seguros_valores_tipo_producto = np.argmax(logits_seguros_valores_tipo_producto, axis=1).flatten()

            # Move logits and labels to CPU
            logits_seguros_valores_tipo_materia = logits_seguros_valores_tipo_materia.detach().cpu().numpy()
            pred_seguros_valores_tipo_materia = np.argmax(logits_seguros_valores_tipo_materia, axis=1).flatten()

            # Move logits and labels to CPU
            logits_seguros_valores_nombre_entidad = logits_seguros_valores_nombre_entidad.detach().cpu().numpy()
            pred_seguros_valores_nombre_entidad = np.argmax(logits_seguros_valores_nombre_entidad, axis=1).flatten()
            


            
            # Tipo Mercado

            tipo_mercado_opciones = open('models/tipo_mercado_target_names.txt', 'r').read().splitlines()
            tipo_mercado_opciones = {v: k for v, k in enumerate(tipo_mercado_opciones)}
            
            
            
        
            sentence_data.append(sentences[i])
            
            labels = {}
            labels['sentence'] = sentences[i]

            if int(pred_tipo_mercado)==0:
                sentence_data.append(tipo_mercado_opciones[int(pred_tipo_mercado)])
                labels['Tipo_Mercado'] = tipo_mercado_opciones[int(pred_tipo_mercado)]

                # Bancos Tipo entidad
                bancos_tipo_entidad_opciones = open('models/bancos_tipo_entidad_target_names.txt', 'r').read().splitlines()
                bancos_tipo_entidad_opciones = {v: k for v, k in enumerate(bancos_tipo_entidad_opciones)}
                sentence_data.append(bancos_tipo_entidad_opciones[int(pred_bancos_tipo_entidad)])
                
                labels['Tipo_Entidad'] = bancos_tipo_entidad_opciones[int(pred_bancos_tipo_entidad)]

                # Bancos nombre Entidad
                bancos_nombre_entidad_opciones = open('models/bancos_nombre_entidad_target_names.txt', 'r').read().splitlines()
                bancos_nombre_entidad_opciones = {v: k for v, k in enumerate(bancos_nombre_entidad_opciones)}
                sentence_data.append(bancos_nombre_entidad_opciones[int(pred_bancos_nombre_entidad)])
         
                
                labels['Nombre_Entidad'] = bancos_nombre_entidad_opciones[int(pred_bancos_nombre_entidad)]

                # Bancos tipo materia
                bancos_tipo_materia_opciones = open('models/bancos_tipo_materia_target_names.txt', 'r').read().splitlines()
                bancos_tipo_materia_opciones = {v: k for v, k in enumerate(bancos_tipo_materia_opciones)}
                sentence_data.append(bancos_tipo_materia_opciones[int(pred_bancos_tipo_materia)])
  
                
                labels['Tipo_Materia'] = bancos_tipo_materia_opciones[int(pred_bancos_tipo_materia)]

                # Bancos tipo producto
                bancos_tipo_producto_opciones = open('models/bancos_tipo_producto_target_names.txt', 'r').read().splitlines()
                bancos_tipo_producto_opciones = {v: k for v, k in enumerate(bancos_tipo_producto_opciones)}
                sentence_data.append(bancos_tipo_producto_opciones[int(pred_bancos_tipo_producto)])
                
                labels['Tipo_Producto'] = bancos_tipo_producto_opciones[int(pred_bancos_tipo_producto)]

            else:
 
                tipo_mercado_seguros_valores_opciones = open('models/tipo_mercado_seguros_valores_target_names.txt', 'r').read().splitlines()
                tipo_mercado_seguros_valores_opciones = {v: k for v, k in enumerate(tipo_mercado_seguros_valores_opciones)}
                sentence_data.append(tipo_mercado_seguros_valores_opciones[int(pred_tipo_mercado_seguros_valores)])
                
                labels['Tipo_Mercado'] = tipo_mercado_seguros_valores_opciones[int(pred_tipo_mercado_seguros_valores)]

                # Seguros Valores Tipo entidad
                seguros_valores_tipo_entidad_opciones = open('models/seguros_valores_tipo_entidad_target_names.txt', 'r').read().splitlines()
                seguros_valores_tipo_entidad_opciones = {v: k for v, k in enumerate(seguros_valores_tipo_entidad_opciones)}
                sentence_data.append(seguros_valores_tipo_entidad_opciones[int(pred_seguros_valores_tipo_entidad)])
                
                labels['Tipo_Entidad'] = seguros_valores_tipo_entidad_opciones[int(pred_seguros_valores_tipo_entidad)]

                # Seguros Valores nombre Entidad
                seguros_valores_nombre_entidad_opciones = open('models/seguros_valores_nombre_entidad_target_names.txt', 'r').read().splitlines()
                seguros_valores_nombre_entidad_opciones = {v: k for v, k in enumerate(seguros_valores_nombre_entidad_opciones)}
                sentence_data.append(seguros_valores_nombre_entidad_opciones[int(pred_seguros_valores_nombre_entidad)])
                
                labels['Nombre_Entidad'] = seguros_valores_nombre_entidad_opciones[int(pred_seguros_valores_nombre_entidad)]

                # Seguros Valores tipo materia
                seguros_valores_tipo_materia_opciones = open('models/seguros_valores_tipo_materia_target_names.txt', 'r').read().splitlines()
                seguros_valores_tipo_materia_opciones = {v: k for v, k in enumerate(seguros_valores_tipo_materia_opciones)}
                sentence_data.append(seguros_valores_tipo_materia_opciones[int(pred_seguros_valores_tipo_materia)])
                
                labels['Tipo_Materia'] = seguros_valores_tipo_materia_opciones[int(pred_seguros_valores_tipo_materia)]

                # Seguros Valores tipo producto
                seguros_valores_tipo_producto_opciones = open('models/seguros_valores_tipo_producto_target_names.txt', 'r').read().splitlines()
                seguros_valores_tipo_producto_opciones = {v: k for v, k in enumerate(seguros_valores_tipo_producto_opciones)}
                sentence_data.append(seguros_valores_tipo_producto_opciones[int(pred_seguros_valores_tipo_producto)])
                
                labels['Tipo_Producto'] = seguros_valores_tipo_producto_opciones[int(pred_seguros_valores_tipo_producto)]

            predictions.append(labels)
            table.append(sentence_data)
            

        return predictions

class DummyClassifier:
    def __init__(self, classes: list) -> str:
        classes = self.classes
    def predict(self, x: list):
        length = len(x)
        predictions = [random.choice(self.classes) for _ in range(length)]
        return predictions

class DummyMercadoClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["seguros", "bancos", "valores"]
        
class DummyTipoEntidadClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["entidad_a", "entidad_b", "entidad_c"]

class DummyNombreEntidadClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["entidad_A", "entidad_B", "entidad_C"]

class DummyTipoProductoClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["producto_a", "producto_b", "producto_c"]

class DummyTipoMateriaClassifier(DummyClassifier):
    def __init__(self) -> str:
        self.classes = ["materia_a", "materia_b", "materia_c"]
              
if __name__=='__main__':
    c = ClaimClassifier()
    c.predict(['reclamo por seguro de vida'])