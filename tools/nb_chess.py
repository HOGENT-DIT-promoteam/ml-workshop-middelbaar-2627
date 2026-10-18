import sys
from nbgen import build

REPO = "HoGentTIN/ml-workshop-middelbaar-2627"
NAME = "CNN_Schaakstukken"

cells = [
("md", f"""
<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{NAME}.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ♟️ Bonus: een schaakbord lezen met een CNN

Je hebt in de vorige notebook een CNN gebouwd voor steen, papier en schaar. Nu pakken we iets moeilijkers aan:
we geven de computer een **foto van een schaakbord** en laten hem uitschrijven **welk stuk op welk vakje staat**.

**Het plan:**
1. Een schaakbord bestaat uit 8 × 8 = **64 vakjes**. We knippen elk bord in 64 kleine afbeeldingen.
2. We trainen een CNN dat voor één vakje zegt wat erop staat: **leeg**, of één van de **12 stukken** (6 wit + 6 zwart). Dat zijn **13 klassen**.
3. We laten het CNN alle 64 vakjes van een bord bekijken en zetten de antwoorden weer samen tot een volledige stelling.

Deze keer krijg je minder hulp: in deel C bouw je het CNN **helemaal zelf**!

⚡ Zet eerst een GPU aan: **Runtime → Runtimetype wijzigen → T4 GPU**.
"""),
("code", """
import os, zipfile, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers
"""),

# ---------------- Deel A
("md", """
---
## Deel A: De data

De afbeeldingen komen uit de dataset [Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions) op Kaggle.
We gebruiken een kleine selectie: **500 borden** om te trainen en **100 borden** om te testen.
"""),
("code", f"""
URL = "https://raw.githubusercontent.com/{REPO}/main/data/schaakborden.zip"

if not os.path.exists("schaakborden"):
    if not os.path.exists("schaakborden.zip"):
        urllib.request.urlretrieve(URL, "schaakborden.zip")
    zipfile.ZipFile("schaakborden.zip").extractall(".")

train_bestanden = sorted(os.listdir("schaakborden/train"))
test_bestanden = sorted(os.listdir("schaakborden/test"))
print(len(train_bestanden), "trainborden en", len(test_bestanden), "testborden")
print("Voorbeeld van een bestandsnaam:", train_bestanden[0])
"""),
("md", """
### De FEN-notatie

De **bestandsnaam** van elke afbeelding vertelt welke stukken er op het bord staan, in de zogenaamde **FEN-notatie**:
- Het bord wordt rij per rij beschreven, **van boven naar onder**. De rijen zijn gescheiden door een `-`.
- Elke letter is een stuk: **K**oning, **Q** = dame, **R** = toren, **B** = loper, **N** = paard, **P** = pion.
- **Hoofdletter** = wit stuk, **kleine letter** = zwart stuk.
- Een **cijfer** geeft aan hoeveel **lege vakjes** er na elkaar komen.

Bv. de rij `1b1B1b2` betekent: leeg, zwarte loper, leeg, witte loper, leeg, zwarte loper, leeg, leeg.
"""),
("code", """
def toon_bord(bestand, map="schaakborden/train"):
    plt.figure(figsize=(5, 5))
    plt.imshow(Image.open(os.path.join(map, bestand)))
    plt.title(bestand.replace(".jpeg", ""), fontsize=9)
    plt.axis("off")
    plt.show()

toon_bord(train_bestanden[0])
"""),
("md", """
✏️ **Opdracht:** controleer met de afbeelding hierboven of de **eerste rij** van de FEN klopt.

Nu schrijven we een functie die een FEN omzet naar een lijst van 8 rijen met elk 8 vakjes.
Een leeg vakje stellen we voor met een punt `.`.

✏️ **Opdracht:** vul de functie aan. Als het teken een cijfer is, moet je zoveel keer een `"."` toevoegen.
(Tip: `["."] * 3` geeft `[".", ".", "."]`.)
"""),
("code", """
def fen_naar_bord(fen):
    bord = []
    for rij in fen.split("-"):
        vakjes = []
        for teken in rij:
            if teken.isdigit():
                # >>> TODO: voeg int(teken) keer een "." toe aan vakjes
                vakjes.extend(["."] * int(teken))
                # ---
                vakjes.extend(...)
                # <<<
            else:
                vakjes.append(teken)
        bord.append(vakjes)
    return bord

for rij in fen_naar_bord(train_bestanden[0].replace(".jpeg", "")):
    print(" ".join(rij))
"""),

# ---------------- Deel B
("md", """
---
## Deel B: Het bord in 64 vakjes knippen

Elke afbeelding is 400 × 400 pixels. We maken ze **zwart-wit** en verkleinen ze naar 200 × 200 pixels:
zo is elk vakje **25 × 25** pixels. (Kleur hebben we niet nodig: witte en zwarte stukken zie je ook in grijstinten.)

Een afbeelding is een tabel met getallen, dus een vakje uitknippen is gewoon een stukje van de tabel nemen.
Voor het vakje op rij `r` en kolom `k` (beginnend bij 0) heb je:
- de pixelrijen van `r * 25` tot `(r + 1) * 25`
- de pixelkolommen van `k * 25` tot `(k + 1) * 25`

✏️ **Opdracht:** vul de code aan om het juiste vakje uit te knippen.
"""),
("code", """
VAKJE = 25
KLASSEN = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
NAMEN = ["leeg", "w pion", "w paard", "w loper", "w toren", "w dame", "w koning",
         "z pion", "z paard", "z loper", "z toren", "z dame", "z koning"]

def laad_bord(pad):
    bord = Image.open(pad).convert("L").resize((8 * VAKJE, 8 * VAKJE))
    return np.array(bord) / 255.0

def knip_vakjes(bord):
    vakjes = []
    for r in range(8):
        for k in range(8):
            # >>> TODO: knip het vakje op rij r en kolom k uit
            vakje = bord[r * VAKJE:(r + 1) * VAKJE, k * VAKJE:(k + 1) * VAKJE]
            # ---
            vakje = bord[... : ..., ... : ...]
            # <<<
            vakjes.append(vakje)
    return np.array(vakjes)

bord = laad_bord(os.path.join("schaakborden/train", train_bestanden[0]))
vakjes = knip_vakjes(bord)
labels = np.array(fen_naar_bord(train_bestanden[0].replace(".jpeg", ""))).flatten()
print("Vorm van de vakjes:", vakjes.shape)

plt.figure(figsize=(7, 7.5))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(vakjes[i], cmap="gray", vmin=0, vmax=1)
    plt.title(labels[i], fontsize=8)
    plt.axis("off")
plt.show()
"""),
("md", """
Klopt het? Dan doen we dit nu voor **alle** borden. Zo krijgen we 500 × 64 = 32 000 trainingsvakjes.
Elk label zetten we om naar een getal: de positie in de lijst `KLASSEN`.
"""),
("code", """
def maak_dataset(map, bestanden):
    X, y = [], []
    for bestand in bestanden:
        X.append(knip_vakjes(laad_bord(os.path.join(map, bestand))))
        bord = fen_naar_bord(bestand.replace(".jpeg", ""))
        y.extend(KLASSEN.index(stuk) for rij in bord for stuk in rij)
    return np.concatenate(X)[..., None], np.array(y)

X_train, y_train = maak_dataset("schaakborden/train", train_bestanden)
X_test, y_test = maak_dataset("schaakborden/test", test_bestanden)
print("Training:", X_train.shape, " Test:", X_test.shape)
"""),
("code", """
aantallen = np.bincount(y_train, minlength=len(KLASSEN))
plt.figure(figsize=(10, 3))
plt.bar(NAMEN, aantallen)
plt.xticks(rotation=45, ha="right")
plt.ylabel("aantal vakjes")
plt.show()
print(f"{aantallen[0] / len(y_train):.0%} van de vakjes is leeg")
"""),
("md", """
❓ **Vraag:** stel dat een (heel dom) model **altijd** "leeg" zegt. Welke accuracy haalt dat model?
Wat zegt dat over hoe je de accuracy van ons echte model moet beoordelen?
"""),

# ---------------- Deel C
("md", """
---
## Deel C: Bouw zelf een CNN 🛠️

✏️ **Opdracht:** bouw een CNN voor de vakjes. Denk aan de bouwstenen uit de slides en de vorige notebook:
- De invoer is een vakje van **25 × 25 × 1** (zwart-wit, dus maar 1 kanaal).
- Een paar blokken `Conv2D` + `MaxPooling2D` om kenmerken te zoeken. (Tip: gebruik `padding="same"` zodat de kleine vakjes niet te snel krimpen.)
- `Flatten` en een `Dense`-laag om te beslissen.
- Een uitvoerlaag met **evenveel neuronen als er klassen zijn** en activatie `softmax`.
"""),
("code", """
model = keras.Sequential([
    keras.Input(shape=(VAKJE, VAKJE, 1)),
    # >>> TODO: voeg hier je lagen toe
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(KLASSEN), activation="softmax"),
    # <<<
])
model.summary()
"""),
("code", """
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
geschiedenis = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
"""),

# ---------------- Deel D
("md", """
---
## Deel D: Evalueren per vakje
"""),
("code", """
_, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy per vakje op de testset: {acc:.2%}")

voorspeld = model.predict(X_test, verbose=0).argmax(axis=1)

from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay.from_predictions(y_test, voorspeld, labels=range(len(KLASSEN)),
                                        display_labels=NAMEN, cmap="Blues", ax=ax, colorbar=False)
plt.xticks(rotation=45, ha="right")
plt.show()
"""),
("md", """
Welke vakjes gingen fout? (Als er geen fouten zijn: proficiat! 🎉)
"""),
("code", """
fout = np.where(voorspeld != y_test)[0]
print(len(fout), "fouten op", len(y_test), "vakjes")

plt.figure(figsize=(12, 3))
for n, i in enumerate(fout[:10]):
    plt.subplot(1, 10, n + 1)
    plt.imshow(X_test[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
    plt.title(f"echt: {NAMEN[y_test[i]]}\\nvoorspeld:\\n{NAMEN[voorspeld[i]]}", fontsize=8, color="red")
    plt.axis("off")
plt.show()
"""),

# ---------------- Deel E
("md", """
---
## Deel E: Een volledig bord lezen ♟️

Nu zetten we alles samen: we knippen een testbord in 64 vakjes, laten het CNN elk vakje voorspellen en zetten
de antwoorden weer om naar een FEN. De functie `bord_naar_fen` doet het omgekeerde van `fen_naar_bord`:
lege vakjes na elkaar worden weer een cijfer.
"""),
("code", """
def bord_naar_fen(stukken):
    rijen = []
    for r in range(8):
        rij, leeg = "", 0
        for stuk in stukken[r * 8:(r + 1) * 8]:
            if stuk == ".":
                leeg += 1
            else:
                rij += (str(leeg) if leeg else "") + stuk
                leeg = 0
        rijen.append(rij + (str(leeg) if leeg else ""))
    return "-".join(rijen)

def lees_bord(bestand, map="schaakborden/test"):
    vakjes = knip_vakjes(laad_bord(os.path.join(map, bestand)))[..., None]
    klassen = model.predict(vakjes, verbose=0).argmax(axis=1)
    return bord_naar_fen([KLASSEN[c] for c in klassen])

bestand = test_bestanden[0]
echt = bestand.replace(".jpeg", "")
gelezen = lees_bord(bestand)

toon_bord(bestand, "schaakborden/test")
print("Echte FEN:    ", echt)
print("Gelezen FEN:  ", gelezen)
print("Juist!" if gelezen == echt else "Niet helemaal juist...")
print("Bekijk de stelling op Lichess: https://lichess.org/editor/" + gelezen.replace("-", "/"))
"""),
("md", """
❓ **Vraag:** stel dat je model **99,5%** van de vakjes juist heeft. Wat is dan de kans dat **alle 64** vakjes van een bord juist zijn?
(Tip: $0{,}995^{64}$.) Bereken het en vergelijk met het resultaat hieronder.

✏️ **Opdracht:** bereken voor hoeveel testborden het model de **volledige** FEN juist heeft.
"""),
("code", """
juist = 0
for bestand in test_bestanden:
    echt = bestand.replace(".jpeg", "")
    # >>> TODO: verhoog juist met 1 als lees_bord(bestand) gelijk is aan de echte FEN
    if lees_bord(bestand) == echt:
        juist += 1
    # ---
    if ...:
        juist += 1
    # <<<
print(f"{juist} van de {len(test_bestanden)} borden volledig juist ({juist / len(test_bestanden):.0%})")
"""),
("md", """
---
## 🏆 Uitdaging

- Haal je **100%** van de borden juist? Probeer een groter netwerk, meer epochs, of `layers.Dropout`.
- Wat gebeurt er als je de vakjes **kleiner** maakt (bv. `VAKJE = 16`)? Wanneer kan het netwerk de stukken niet meer van elkaar onderscheiden?
- Zoek online een screenshot van een schaakbord (bv. van Lichess of Chess.com), knip het bij tot net het bord, en laat je model het lezen.
  Werkt dat? Waarom (niet)?
"""),
]

build(cells, sys.argv[1] + f"/{NAME}.ipynb", sys.argv[2] + f"/{NAME}_oplossing.ipynb")
