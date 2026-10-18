import sys
from nbgen import build

REPO = "HoGentTIN/ml-workshop-middelbaar-2627"
NAME = "CNN_MNIST"

cells = [
("md", f"""
<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{NAME}.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ✍️ Van ENNUI naar Python: handgeschreven cijfers herkennen

In [ENNUI](https://math.mit.edu/ennui/) heb je een CNN in elkaar gesleept en getraind op **MNIST**:
70 000 handgeschreven cijfers van 28 × 28 pixels.
In deze notebook laten we datzelfde netwerk draaien in **Python**, met de bibliotheek **Keras**.

| Deel | Onderwerp |
|---|---|
| A | Je ENNUI-export laten draaien |
| B | Hetzelfde CNN, stap voor stap in Keras |
| C | Van cijfers naar steen, papier, schaar |

Voer de cellen uit met **Shift + Enter**. In cellen met `# TODO` vul je zelf code aan op de plaats van de `...`.
"""),

# ---------------- Deel A
("md", """
---
## Deel A: Je ENNUI-export

1. Klik in ENNUI rechts op **Export to Python** en kopieer de code.
2. Plak de code in de cel hieronder (vervang de commentaarregel) en voer de cel uit.
"""),
("code", """
# >>> Plak hier de code uit ENNUI (Export to Python)
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Concatenate, BatchNormalization, Add
from keras.layers import Conv2D, MaxPooling2D, ReLU
from keras import backend as K

batch_size = 64
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols, channels = 28, 28, 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

############################# Architecture made by Ennui
x0 = Input(shape=input_shape)
x1 = Conv2D(16, (3,3),strides=(1,1), activation='relu', padding='same')(x0)
x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x1)
x3 = Conv2D(32, (3,3),strides=(1,1), activation='relu', padding='same')(x2)
x4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x3)
x5 = Flatten()(x4)
x6 = Dense(64, activation='relu')(x5)
x7 = Dense(10, activation='softmax')(x6)
model = Model(inputs=x0, outputs=x7)
#############################

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# ---
# <<<
"""),
("md", """
### 🐞 Foutmelding?

Krijg je deze fout?

```
ValueError: Argument(s) not recognized: {'lr': 0.01}
```

ENNUI is gemaakt voor een **oudere versie** van Keras. In de nieuwe versie heet de parameter voor de leersnelheid
niet meer `lr` maar `learning_rate`.

✏️ **Opdracht:** zoek in je code de regel met `optimizer=` en vervang `lr=` door `learning_rate=`. Voer de cel opnieuw uit.

💡 Programmeurs komen zulke fouten elke dag tegen. De laatste regel van de foutmelding vertelt meestal precies wat er mis is.

❓ **Vragen:**
1. Haal je in Python ongeveer dezelfde accuracy als in ENNUI?
2. Zoek in de code het stuk tussen de `#####`-lijnen. Herken je de lagen die je in ENNUI hebt gesleept?
"""),

# ---------------- Deel B
("md", """
---
## Deel B: Hetzelfde CNN, stap voor stap

De code van ENNUI werkt, maar ze is niet zo makkelijk te lezen. We bouwen nu hetzelfde netwerk opnieuw op, stap voor stap.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Trainingsset:", x_train.shape, " Testset:", x_test.shape)
"""),
("code", """
plt.figure(figsize=(10, 4))
for i in range(20):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")
plt.show()
"""),
("md", """
Elke afbeelding is een tabel van **28 × 28** getallen van 0 (zwart) tot 255 (wit).
Een CNN verwacht ook een **kanaal-as**: zwart-wit heeft 1 kanaal, kleur heeft er 3 (RGB).
Dus van `(28, 28)` maken we `(28, 28, 1)`.

✏️ **Opdracht:** normaliseer de pixelwaarden naar het bereik 0–1.
"""),
("code", """
# >>> TODO: deel door 255 zodat alle waarden tussen 0 en 1 liggen
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0
# ---
x_train = x_train[..., None] / ...
x_test = x_test[..., None] / ...
# <<<
print("Nieuwe vorm:", x_train.shape, " min:", x_train.min(), " max:", x_train.max())
"""),
("md", """
### Het model

In ENNUI verbond je de lagen met pijlen. In Keras zet je ze gewoon **onder elkaar** in een `Sequential`-model.

| ENNUI | Keras |
|---|---|
| Input | `keras.Input(shape=(28, 28, 1))` |
| Convolution + ReLU | `layers.Conv2D(16, 3, activation="relu", padding="same")` |
| Max Pooling | `layers.MaxPooling2D()` |
| Flatten | `layers.Flatten()` |
| Dense + ReLU | `layers.Dense(64, activation="relu")` |
| Output | `layers.Dense(10, activation="softmax")` |

✏️ **Opdracht:** maak het model af met dezelfde lagen als in je ENNUI-netwerk:
een tweede `Conv2D` met 32 filters + `MaxPooling2D`, daarna `Flatten`, `Dense(64)` en de uitvoerlaag.
"""),
("code", """
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    # >>> TODO: voeg de overige lagen toe
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
    # <<<
])
model.summary()
"""),
("md", """
❓ Kijk naar `Output Shape`: hoe groot zijn de feature maps na elke laag? Klopt dat met wat ENNUI toonde?

### Trainen

Twee kleine verschillen met de ENNUI-code:
- `adam` is een slimmere versie van `SGD`: hij leert meestal sneller.
- Met `sparse_categorical_crossentropy` mogen de labels gewoon getallen (0–9) blijven. Dan is `to_categorical` niet nodig.
"""),
("code", """
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
geschiedenis = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))
"""),
("code", """
plt.figure(figsize=(6, 4))
plt.plot(geschiedenis.history["accuracy"], "o-", label="training")
plt.plot(geschiedenis.history["val_accuracy"], "o-", label="test")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(alpha=0.3)
plt.show()

_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy op de testset: {acc:.2%}")
"""),
("md", """
### Welke cijfers gaan fout?
"""),
("code", """
voorspeld = model.predict(x_test, verbose=0).argmax(axis=1)
fout = np.where(voorspeld != y_test)[0]
print(len(fout), "fouten op", len(y_test), "testafbeeldingen")

plt.figure(figsize=(12, 3))
for n, i in enumerate(fout[:10]):
    plt.subplot(1, 10, n + 1)
    plt.imshow(x_test[i, :, :, 0], cmap="gray")
    plt.title(f"echt {y_test[i]}\\nzegt {voorspeld[i]}", color="red", fontsize=9)
    plt.axis("off")
plt.show()
"""),
("code", """
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, voorspeld, cmap="Blues", colorbar=False)
plt.show()
"""),
("md", """
❓ Welke cijfers worden het vaakst met elkaar verward? Zou jij die fouten ook maken?
"""),

# ---------------- Deel C
("md", """
---
## Deel C: Van cijfers naar steen, papier, schaar ✊✋✌️

Hetzelfde recept werkt voor heel andere afbeeldingen. Dit verandert er:

| | MNIST | Steen, papier, schaar |
|---|---|---|
| Afbeeldingen | 28 × 28, **zwart-wit** | 100 × 100, **kleur** |
| Input | `keras.Input(shape=(28, 28, 1))` | `keras.Input(shape=(100, 100, 3))` |
| Aantal klassen | 10 | 3 |
| Uitvoerlaag | `layers.Dense(10, activation="softmax")` | `layers.Dense(3, activation="softmax")` |
| Data | 70 000 afbeeldingen | slechts 2 892 afbeeldingen ⇒ gevaar voor **overfitting** |

👉 Ga verder met de notebook **`CNN_SteenPapierSchaar.ipynb`**.
"""),
]

build(cells, sys.argv[1] + f"/{NAME}.ipynb", sys.argv[2] + f"/{NAME}_oplossing.ipynb")
