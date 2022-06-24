# Importing essential libraries
from hashlib import new
from flask import Flask, render_template, request, send_file
import pickle
import tensorflow as tf
import numpy as np
from yaml import dump
from differential_evolution import differential_evolution
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64


def perturb_image(xs, img):
    # Controllo se il numero di pixel perturbati è pari solo a uno e in tal caso
    # lo trasformo in una lista per lasciare inalterato il calcolo
    if xs.ndim < 2:
        xs = np.array([xs])
       
    #Copio l'immagine immagine  n == len(xs) volte per creare n immagini perturbate(inutilizzato)
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
   
    # La perturbazione deve essere intera
    xs = xs.astype(int)
    
    for x,img in zip(xs, imgs): #ottengo (vettore_perturbazioni x - immagine da perturbare(qui una))

        # Split di x in un'array di 3 elementi ciascuno, ossia per la singola perturbazione 
        # i.e [[x,y,z], ...]
        pixels = np.split(x, len(x) // 3) 
        for pixel in pixels:
            #A ogni pixel (x,y) assegno il suo livello di grigio z
            x_pos, y_pos, *gray = pixel  #[x,y,z]
            img[x_pos, y_pos] = gray
    
    return imgs

def predict_classes(xs, img, target_class, model):
    imgs_perturbed = perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:,target_class]
    return predictions 
def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    attack_image = perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)

    if verbose:
        print('Confidence:', round(confidence[target_class],ndigits=3))
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):  
        return True 

def attack(classe,img, model, target=None, pixel_count=1, 
           maxiter=75, popsize=400, verbose=False):
    # Seleziono la classe target a seconda che l'attacco sia targeted o untargeted 
    targeted_attack = target is not None
    target_class = target if targeted_attack else int(classe)
    
    # Definisco vettore perturbazione
    bounds = [(0,28), (0,28), (0,2)] * pixel_count
    
    # Moltiplicatore della popolazione a seconda della dimensione del vettore di perturbazione
    popmul = max(1, popsize // len(bounds))
    
    img = np.reshape(img,(1,28,28,1))
    
    # Funzioni ausiliarie per l'algoritmo evoluzione differenziale
    def predict_fn(xs): #nella forma f(x,args) dove x è un vettore 1d 
        return predict_classes(xs, img, target_class, 
                               model)
    
    def callback_fn(x, convergence): #quando ritorna True l'algoritmo termina
        return attack_success(x, img, target_class, 
                              model, targeted_attack, verbose)
    
    #Differential Evolution
    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize= popmul,mutation=1,recombination = 1, atol = -1, callback=callback_fn, polish=False)

    
    # Calcolo statistiche
    attack_image = perturb_image(attack_result.x, img)[0]
    prior_probs = model.predict(img)
    
    attack_image = np.reshape(attack_image,(1,28,28,1))
    predicted_probs = model.predict(attack_image)
  
    predicted_class = np.argmax(predicted_probs)
    actual_class = int(classe)
    success = predicted_class != actual_class
    
    return [attack_image, pixel_count, classe, actual_class, predicted_class, success, prior_probs, predicted_probs, attack_result.x]
   

model = tf.keras.models.load_model('/Users/raffaelerusso/Desktop/Uni/ro/RO-progetto/ModelloPesi/mnist_model.h5')
model.load_weights('/Users/raffaelerusso/Desktop/Uni/ro/RO-progetto/ModelloPesi/mnist_weights.h5')
 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
      pixel = int(request.form["pixels"])
      f = request.files['file']
      f.save(f.filename)
      pic = Image.open(f)
      pic = np.float32(np.reshape(pic,(1,28,28,1)))
      target = np.argmax(model.predict(pic),axis = 1)
      result =  attack(target, pic,model, pixel_count=pixel,maxiter=100,popsize=100,target = None, verbose=False)
      img_att = np.uint8(result[0])*255
      img_att = np.reshape(img_att,(28,28))
      #return "Predetta: "+str(target)+" Post attacco: "+(str(result[4]))
      
      img = Image.fromarray(img_att)

        # create file-object in memory
      file_object = io.BytesIO()

       # write PNG in file-object
      img.save(file_object, 'PNG',quality = 100)
      encoded_img_data = base64.b64encode(file_object.getvalue())

       # move to beginning of file so `send_file()` it will read from start    
      file_object.seek(0)

      

      return render_template('result.html', img_data=encoded_img_data.decode('utf-8') , prediction=[int(target),int(result[4])])
   
if __name__ == '__main__':
    app.run(debug=True)