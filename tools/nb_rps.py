import sys
from nbgen import build

REPO = "HoGentTIN/ml-workshop-middelbaar-2627"
NAME = "CNN_SteenPapierSchaar"

cells = [
("md", f"""
<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{NAME}.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ✊✋✌️ Steen, Papier, Schaar met een Convolutioneel Neuraal Netwerk

In ENNUI en de MNIST-notebook heb je een **Convolutioneel Neuraal Netwerk (CNN)** gebouwd dat handgeschreven cijfers herkent.
Nu gebruiken we hetzelfde recept voor echte kleurenfoto's: we leren een computer het verschil zien tussen
**steen**, **papier** en **schaar**.

**Wat ga je doen?**

| Deel | Onderwerp |
|---|---|
| A | De data bekijken en klaarmaken |
| B | Zelf een convolutie (filter) uitproberen |
| C | Een "gewoon" neuraal netwerk als vergelijkingspunt |
| D | Zelf een CNN bouwen en trainen |
| E | Het model evalueren |
| F | In het netwerk kijken: wat heeft het geleerd? |
| G | Uitdaging: maak het model beter & test met je eigen hand! |

**Hoe werk je?**
- Voer de cellen één voor één uit met **Shift + Enter**.
- In cellen met `# TODO` moet je zelf code aanvullen op de plaats van de `...`.
- ⚡ Zet eerst een GPU aan: menu **Runtime → Runtimetype wijzigen → T4 GPU**. Dan gaat trainen veel sneller.
"""),
("code", """
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers

print("TensorFlow versie:", tf.__version__)
print("GPU beschikbaar:", len(tf.config.list_physical_devices("GPU")) > 0)
"""),

# ---------------- Deel A
("md", """
---
## Deel A: De data

We gebruiken de dataset **Rock Paper Scissors** van Google: zo'n 2900 afbeeldingen van handen die steen, papier of schaar tonen.
De dataset is al opgesplitst in:
- een **trainingsset** (2520 afbeeldingen) waarmee het netwerk leert;
- een **testset** (372 afbeeldingen) die het netwerk nooit gezien heeft. Daarmee controleren we achteraf hoe goed het écht is.
"""),
("code", """
(train_ds, test_ds), info = tfds.load(
    "rock_paper_scissors",
    split=["train", "test"],
    as_supervised=True,   # geeft telkens (afbeelding, label)
    with_info=True,
)

KLASSEN = ["steen", "papier", "schaar"]
print(info.splits)
"""),
("md", """
Laten we een paar afbeeldingen bekijken. Het **label** is een getal: `0` = steen, `1` = papier, `2` = schaar.
"""),
("code", """
plt.figure(figsize=(10, 6))
for i, (afbeelding, label) in enumerate(train_ds.take(12)):
    plt.subplot(3, 4, i + 1)
    plt.imshow(afbeelding)
    plt.title(f"{int(label)} = {KLASSEN[int(label)]}")
    plt.axis("off")
plt.show()
"""),
("md", """
### Een afbeelding = een tabel met getallen

Voor een computer is een kleurenafbeelding een 3D-tabel (een *tensor*) met:
**hoogte × breedte × 3 kleurkanalen** (rood, groen, blauw).
Elk getal ligt tussen **0** (geen licht) en **255** (maximaal licht).

✏️ **Opdracht:** vraag hieronder de `shape` (vorm) op van één afbeelding, en de kleinste en grootste waarde.
"""),
("code", """
afbeelding, label = next(iter(train_ds))

# >>> TODO: print de vorm (shape) van de afbeelding
print("Vorm:", afbeelding.shape)
# ---
print("Vorm:", ...)
# <<<
print("Kleinste waarde:", afbeelding.numpy().min())
print("Grootste waarde:", afbeelding.numpy().max())
print("Pixel linksboven (R, G, B):", afbeelding[0, 0].numpy())
"""),
("md", """
❓ **Vraag:** hoeveel getallen zitten er in totaal in één afbeelding?

### Data klaarmaken

Twee stappen:
1. **Verkleinen** naar 100 × 100 pixels: dat traint veel sneller en we verliezen weinig informatie.
2. **Normaliseren**: neurale netwerken leren beter met kleine getallen. We zetten de waarden om van 0–255 naar **0–1**.

Daarna verdelen we de data in **batches** van 32 afbeeldingen: het netwerk bekijkt telkens 32 afbeeldingen tegelijk.

✏️ **Opdracht:** vul de normalisatie aan.
"""),
("code", """
GROOTTE = 100

def klaarmaken(afbeelding, label):
    afbeelding = tf.image.resize(afbeelding, (GROOTTE, GROOTTE))
    # >>> TODO: deel de pixelwaarden door 255 zodat ze tussen 0 en 1 liggen
    afbeelding = afbeelding / 255.0
    # ---
    afbeelding = ...
    # <<<
    return afbeelding, label

train = train_ds.map(klaarmaken).cache().shuffle(1000, seed=42).batch(32).prefetch(tf.data.AUTOTUNE)
test = test_ds.map(klaarmaken).cache().batch(32).prefetch(tf.data.AUTOTUNE)

x, y = next(iter(train))
print("Vorm van één batch:", x.shape)
print("Waarden tussen", float(tf.reduce_min(x)), "en", float(tf.reduce_max(x)))
"""),

# ---------------- Deel B
("md", """
---
## Deel B: Wat is een convolutie?

Een **convolutie** schuift een klein venster (een **filter** of **kernel**, bv. 3 × 3 getallen) over de afbeelding.
Op elke positie vermenigvuldig je de 9 pixels met de 9 getallen van het filter en tel je alles op.
Zo krijg je een nieuwe afbeelding: een **feature map**.

Afhankelijk van de getallen in het filter detecteert het iets anders: verticale randen, horizontale randen, vervaging, ...

We proberen het uit op een zwart-wit versie van een afbeelding.
"""),
("code", """
def toon_filter(kernel, titel):
    grijs = tf.image.rgb_to_grayscale(x[0:1])                       # 1 afbeelding, zwart-wit
    k = tf.constant(kernel, dtype=tf.float32)[:, :, None, None]      # vorm (3, 3, 1, 1)
    resultaat = tf.nn.conv2d(grijs, k, strides=1, padding="SAME")

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(grijs[0, :, :, 0], cmap="gray"); plt.title("origineel"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(kernel, cmap="bwr", vmin=-2, vmax=2); plt.title("filter")
    for (i, j), v in np.ndenumerate(np.array(kernel)):
        plt.text(j, i, f"{v:g}", ha="center", va="center")
    plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(resultaat[0, :, :, 0], cmap="gray"); plt.title(titel); plt.axis("off")
    plt.show()

verticale_randen = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
toon_filter(verticale_randen, "verticale randen")
"""),
("md", """
✏️ **Opdracht:**
1. Maak hieronder een filter dat **horizontale randen** detecteert. (Tip: draai het filter hierboven een kwartslag.)
2. Probeer daarna ook een **vervagingsfilter**: alle 9 getallen gelijk aan `1/9`.
3. Verzin zelf een filter. Wat gebeurt er?
"""),
("code", """
# >>> TODO: maak een filter voor horizontale randen
horizontale_randen = [[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]]
# ---
horizontale_randen = [[..., ..., ...],
                      [..., ..., ...],
                      [..., ..., ...]]
# <<<
toon_filter(horizontale_randen, "horizontale randen")
"""),
("md", """
💡 **Het grote idee van een CNN:** vroeger bedachten mensen zelf zulke filters. In een CNN **leert het netwerk zelf**
welke filters nuttig zijn! De getallen in de filters zijn gewoon gewichten die tijdens het trainen aangepast worden.
"""),

# ---------------- Deel C
("md", """
---
## Deel C: Eerst een "gewoon" neuraal netwerk

Als vergelijkingspunt trainen we eerst een gewoon (volledig geconnecteerd) neuraal netwerk, zoals in de TensorFlow Playground.
- `Flatten` legt alle 100 × 100 × 3 = 30 000 pixels achter elkaar in één lange rij.
- `Dense(128)` is een laag met 128 neuronen die elk met **alle** 30 000 pixels verbonden zijn.
- `Dense(3, activation="softmax")` geeft voor elke klasse een kans (samen 100%).
"""),
("code", """
dense_model = keras.Sequential([
    keras.Input(shape=(GROOTTE, GROOTTE, 3)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(3, activation="softmax"),
])

dense_model.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
dense_model.summary()
"""),
("code", """
dense_model.fit(train, epochs=5)
_, dense_acc = dense_model.evaluate(test)
print(f"Accuracy op de testset: {dense_acc:.1%}")
"""),
("md", """
📝 **Noteer:** het aantal parameters (`Total params`) en de accuracy op de testset.

❓ Waarom is het eigenlijk geen goed idee om elk neuron met elke pixel te verbinden?
"""),

# ---------------- Deel D
("md", """
---
## Deel D: Een CNN bouwen

Een CNN bestaat uit twee delen:

```
  afbeelding ─▶ [ Conv2D ─▶ MaxPooling ] ─▶ [ Conv2D ─▶ MaxPooling ] ─▶ [ Conv2D ─▶ MaxPooling ] ─▶ Flatten ─▶ Dense ─▶ Dense(3)
               └──────────────────── kenmerken zoeken ─────────────────────────────┘          └─── beslissen ───┘
```

| Laag | Wat doet ze? |
|---|---|
| `Conv2D(16, 3, activation="relu")` | 16 filters van 3 × 3 die elk een kenmerk zoeken (randen, hoeken, ...). `relu` zet negatieve waarden op 0. |
| `MaxPooling2D()` | Neemt van elk blokje 2 × 2 pixels de grootste waarde → de afbeelding wordt 2× kleiner. |
| Dieper in het netwerk | Meer filters (32, 64) die steeds complexere kenmerken herkennen: vingers, vuisten, ... |
| `Flatten` + `Dense` | Combineren alle gevonden kenmerken tot een beslissing. |

✏️ **Opdracht:** maak het model af. De eerste blok staat er al. Voeg toe:
1. een `Conv2D` met **32** filters van 3 × 3 (activatie `relu`) en een `MaxPooling2D`;
2. een `Conv2D` met **64** filters van 3 × 3 (activatie `relu`) en een `MaxPooling2D`;
3. een `Flatten`-laag;
4. een `Dense`-laag met **64** neuronen (activatie `relu`);
5. de uitvoerlaag: `Dense` met **3** neuronen en activatie `softmax`.
"""),
("code", """
cnn = keras.Sequential([
    keras.Input(shape=(GROOTTE, GROOTTE, 3)),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    # >>> TODO: voeg hier de overige lagen toe
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),
    # <<<
])

cnn.summary()
"""),
("md", """
❓ **Vragen bij de samenvatting hierboven:**
1. Kijk naar de kolom `Output Shape`. Wat gebeurt er met de **grootte** van de afbeelding na elke `MaxPooling2D`? En met het **aantal kanalen** na elke `Conv2D`?
2. Vergelijk `Total params` met het gewone netwerk uit deel C. Welk netwerk heeft er meer? Hoe komt dat?
"""),
("md", """
### Trainen

Bij het trainen houden we de testset apart om na elke **epoch** (één keer door alle trainingsafbeeldingen) te controleren hoe goed het model is op nieuwe afbeeldingen.

✏️ **Opdracht:** compileer het model op dezelfde manier als in deel C en train het voor **10** epochs.
"""),
("code", """
# >>> TODO: compileer het CNN (zelfde optimizer, loss en metrics als in deel C)
cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
# <<<

# >>> TODO: vul het aantal epochs in
geschiedenis = cnn.fit(train, epochs=10, validation_data=test)
# ---
geschiedenis = cnn.fit(train, epochs=..., validation_data=test)
# <<<
"""),
("code", """
plt.figure(figsize=(6, 4))
plt.plot(geschiedenis.history["accuracy"], "o-", label="training")
plt.plot(geschiedenis.history["val_accuracy"], "o-", label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
"""),
("md", """
❓ De accuracy op de **training**set wordt (bijna) 100%, maar op de **test**set ligt ze een stuk lager. Hoe noemen we dat?
Waarom is dat een probleem?
"""),

# ---------------- Deel E
("md", """
---
## Deel E: Evalueren

✏️ **Opdracht:** bereken de accuracy van het CNN op de testset met `cnn.evaluate(...)` en vergelijk met het gewone netwerk.
"""),
("code", """
# >>> TODO: evalueer het CNN op de testset
_, cnn_acc = cnn.evaluate(test)
# ---
_, cnn_acc = ...
# <<<
print(f"Gewoon netwerk: {dense_acc:.1%}")
print(f"CNN:            {cnn_acc:.1%}")
"""),
("md", """
Laten we bekijken wat het model voorspelt. Groen = juist, rood = fout.
"""),
("code", """
x_test, y_test = next(iter(test.unbatch().batch(372)))
kansen = cnn.predict(x_test, verbose=0)
voorspeld = kansen.argmax(axis=1)

plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i])
    juist = voorspeld[i] == y_test[i]
    plt.title(f"{KLASSEN[voorspeld[i]]} ({kansen[i].max():.0%})", color="green" if juist else "red")
    plt.axis("off")
plt.show()
"""),
("md", """
Een **confusion matrix** toont welke klassen met elkaar verward worden.
Op de diagonaal staan de juiste voorspellingen; alles daarbuiten is een fout.
"""),
("code", """
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, voorspeld, display_labels=KLASSEN, cmap="Blues")
plt.show()
"""),
("md", """
❓ Welke klassen worden het vaakst met elkaar verward? Kan je een reden bedenken?

En wat met de **fouten**? Hieronder zie je alle afbeeldingen die fout geclassificeerd zijn.
"""),
("code", """
fout = np.where(voorspeld != y_test.numpy())[0]
print(f"{len(fout)} fouten op {len(y_test)} testafbeeldingen")

plt.figure(figsize=(12, 7))
for n, i in enumerate(fout[:15]):
    plt.subplot(3, 5, n + 1)
    plt.imshow(x_test[i])
    plt.title(f"echt: {KLASSEN[y_test[i]]}\\nvoorspeld: {KLASSEN[voorspeld[i]]}", color="red", fontsize=9)
    plt.axis("off")
plt.show()
"""),

# ---------------- Deel F
("md", """
---
## Deel F: In het netwerk kijken 🔍

Wat ziet het netwerk eigenlijk? We tonen de **feature maps** van de eerste convolutielaag:
voor elk van de 16 geleerde filters zie je hoe de afbeelding er na dat filter uitziet.
Vergelijk met de filters die je zelf maakte in deel B!
"""),
("code", """
conv_lagen = [laag for laag in cnn.layers if isinstance(laag, layers.Conv2D)]

def toon_feature_maps(laag_nr, afbeelding):
    kijker = keras.Model(inputs=cnn.inputs, outputs=conv_lagen[laag_nr].output)
    maps = kijker.predict(afbeelding[None], verbose=0)[0]
    aantal = min(maps.shape[-1], 16)

    plt.figure(figsize=(12, 3.5))
    plt.subplot(2, 9, 1); plt.imshow(afbeelding); plt.title("input"); plt.axis("off")
    for k in range(aantal):
        plt.subplot(2, 9, k + 2 + (k >= 8))
        plt.imshow(maps[:, :, k], cmap="viridis")
        plt.axis("off")
    plt.suptitle(f"Conv-laag {laag_nr + 1}: {maps.shape[0]}×{maps.shape[1]} pixels, {maps.shape[-1]} filters (eerste {aantal} getoond)")
    plt.show()

toon_feature_maps(0, x_test[0])
"""),
("md", """
✏️ **Opdracht:** bekijk ook de feature maps van de **derde** convolutielaag (`laag_nr = 2`).
Wat valt je op aan de grootte? Kan je nog herkennen wat er op de afbeelding staat?
"""),
("code", """
# >>> TODO: toon de feature maps van de derde convolutielaag
toon_feature_maps(2, x_test[0])
# ---
toon_feature_maps(..., x_test[0])
# <<<
"""),
("md", """
💡 Hoe dieper in het netwerk, hoe kleiner en abstracter de feature maps. De eerste lagen zien **randen en kleuren**,
de diepere lagen reageren op **vormen** zoals vingers of een vuist. Voor ons zijn die moeilijk te interpreteren, maar voor
de `Dense`-lagen op het einde zijn ze precies de informatie die nodig is om te beslissen.
"""),

# ---------------- Deel G
("md", """
---
## Deel G: Uitdaging 🏆

### G1. Data-augmentatie tegen overfitting

Ons model heeft maar 2520 trainingsafbeeldingen. Een slimme truc: bij elke epoch de afbeeldingen een beetje **willekeurig
aanpassen** (spiegelen, draaien, inzoomen). Zo lijkt het alsof we veel meer verschillende afbeeldingen hebben.
"""),
("code", """
augmentatie = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

plt.figure(figsize=(10, 2.5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(augmentatie(x_test[0:1], training=True)[0])
    plt.axis("off")
plt.suptitle("Dezelfde afbeelding, 5 keer willekeurig aangepast")
plt.show()
"""),
("md", """
✏️ **Opdracht:** bouw een verbeterd CNN en probeer de accuracy op de testset zo hoog mogelijk te krijgen. Ideeën:
- zet `augmentatie` als eerste laag in je model (net na `keras.Input`);
- voeg `layers.Dropout(0.5)` toe vóór de laatste `Dense`-laag (zet tijdens het trainen willekeurig de helft van de neuronen uit);
- meer of minder filters, een extra `Conv2D`-laag, meer epochs, ...

Wie haalt de hoogste test-accuracy? 🥇
"""),
("code", """
mijn_cnn = keras.Sequential([
    keras.Input(shape=(GROOTTE, GROOTTE, 3)),
    # >>> TODO: bouw je eigen, verbeterde CNN
    augmentatie,
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax"),
    # <<<
])

mijn_cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
mijn_geschiedenis = mijn_cnn.fit(train, epochs=15, validation_data=test)

plt.plot(mijn_geschiedenis.history["accuracy"], "o-", label="training")
plt.plot(mijn_geschiedenis.history["val_accuracy"], "o-", label="test")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.3)
plt.show()
"""),
("md", """
### G2. Test met je eigen hand ✋

Neem een foto van je hand (steen, papier of schaar), liefst **tegen een witte achtergrond**, en upload ze hieronder.
"""),
("code", """
from PIL import Image

try:
    from google.colab import files
    upload = files.upload()
    bestanden = list(upload.keys())
except ImportError:
    print("Uploaden werkt enkel in Google Colab.")
    bestanden = []

for bestand in bestanden:
    foto = Image.open(bestand).convert("RGB").resize((GROOTTE, GROOTTE))
    foto = np.array(foto) / 255.0
    kans = mijn_cnn.predict(foto[None], verbose=0)[0]

    plt.imshow(foto)
    plt.title(" | ".join(f"{k}: {p:.0%}" for k, p in zip(KLASSEN, kans)))
    plt.axis("off")
    plt.show()
"""),
("md", """
❓ Werkt het met jouw foto? Kijk nog eens goed naar de trainingsafbeeldingen in deel A: het zijn **computergegenereerde**
handen op een witte achtergrond. Wat betekent dat voor foto's uit de echte wereld?

💡 **Belangrijke les:** een model is maar zo goed als de data waarmee het getraind is!

---
### 🎉 Klaar?

Ga verder met de bonusnotebook **`CNN_Schaakstukken.ipynb`**: daar gebruik je een CNN om automatisch
een schaakbord te "lezen"!
"""),
]

build(cells, sys.argv[1] + f"/{NAME}.ipynb", sys.argv[2] + f"/{NAME}_oplossing.ipynb")
