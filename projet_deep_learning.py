## Projet Deep Learning : Inter classification image
#----------------------------------------------------------------------------------
"""#Etape 0 - Import des librairie"""
#----------------------------------------------------------------------------------
import numpy as np, pandas as pd
print("numpy:", np.__version__, "| pandas:", pd.__version__)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
#----------------------------------------------------------------------------------
DATA_ROOT = Path("/kaggle/input/intel-image-classification")
TRAIN_DIR = DATA_ROOT / "seg_train" / "seg_train"
TEST_DIR  = DATA_ROOT / "seg_test" / "seg_test"
PRED_DIR  = DATA_ROOT / "seg_pred" / "seg_pred"

print("Train exists:", TRAIN_DIR.exists(), "->", TRAIN_DIR)
print("Test exists :", TEST_DIR.exists(), "->", TEST_DIR)
print("Pred exists :", PRED_DIR.exists(), "->", PRED_DIR)

#Vérif des classes dans train
print("Classes dans train:", [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

from pathlib import Path

DATA_ROOT = Path("/kaggle/input/intel-image-classification")
TRAIN_DIR = DATA_ROOT / "seg_train" / "seg_train"
TEST_DIR  = DATA_ROOT / "seg_test" / "seg_test"
PRED_DIR  = DATA_ROOT / "seg_pred" / "seg_pred"

print("Train exists:", TRAIN_DIR.exists(), "->", TRAIN_DIR)
print("Test exists :", TEST_DIR.exists(), "->", TEST_DIR)
print("Pred exists :", PRED_DIR.exists(), "->", PRED_DIR)

#Vérif des classes dans train
print("Classes dans train:", [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

classes = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
print("Classes (6 attendues) :", classes)
#----------------------------------------------------------------------------------
"""#Etape 1 - Introduction et exploration du dataset"""
#----------------------------------------------------------------------------------
#Import des librairies nécessaires
import keras_cv as kcv
from tensorflow.keras import layers

#Paramètres globaux
IMG_SIZE = (150, 150)   #taille des images (hauteur, largeur)
BATCH = 64              #taille des batchs
SEED = 42               #seed pour la reproductibilité

#Chargement du dataset d'entraînement (80% train, 20% val)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='categorical', validation_split=0.2, subset='training', seed=SEED)

#Dataset de validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='categorical', validation_split=0.2, subset='validation', seed=SEED)

#Dataset de test (pas de shuffle pour garder l'ordre)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode='categorical', shuffle=False)

#On récupère les noms des classes détectées
class_names = train_ds.class_names
print("Classes détectées :", class_names)

# --------- PRÉTRAITEMENT ---------
#Normalisation (pixels de 0-255 → 0-1)
normalizer = layers.Rescaling(1./255, name="rescale_01")
normalizer = layers.Rescaling(1./255)

#Data augmentation (appliquée uniquement en entraînement)
augmenter = keras.Sequential([
    layers.RandomContrast(0.30),       #contraste aléatoire
    layers.RandomFlip('horizontal'),   #flip horizontal
    layers.RandomZoom(0.10),           #zoom aléatoire
])

# --------- OPTIMISATION DES PIPELINES I/O ---------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1024).cache().prefetch(AUTOTUNE)  #shuffle + cache + prefetch
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# --------- EXPLORATION RAPIDE ---------
#Visualisation de 9 images augmentées (batch aléatoire du dataset train)
images, labels = next(iter(train_ds))
plt.figure(figsize=(8,6))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    aug = augmenter(images[i], training=True)  #training=True pour forcer les transformations
    plt.imshow(tf.cast(tf.clip_by_value(aug, 0, 255), tf.uint8))
    plt.title(class_names[tf.argmax(labels[i]).numpy()])
    plt.axis('off')
plt.tight_layout(); plt.show()

#--------- TEST NORMALISATION ---------
#Vérification que la normalisation a bien mis les pixels entre 0 et 1
x_batch, _ = next(iter(train_ds))
x_norm = normalizer(x_batch)
print("Normalisation pixels — Avant: min=%.1f max=%.1f | Après: min=%.3f max=%.3f" %
      (float(tf.reduce_min(x_batch)), float(tf.reduce_max(x_batch)),
       float(tf.reduce_min(x_norm)),  float(tf.reduce_max(x_norm))))

#--------- TEST AUGMENTATION ---------
# Visualisation avant/après des augmentations sur 6 images
plt.figure(figsize=(8,8))
for i in range(6):
    #Image originale
    ax = plt.subplot(6,2,2*i+1)
    img_orig = images[i]  # shape (150,150,3)
    plt.imshow(tf.cast(img_orig, tf.uint8)); plt.title("original"); plt.axis("off")

    #image après augmentation (training=True pour appliquer les transfs)
    ax = plt.subplot(6,2,2*i+2)
    img_aug = augmenter(img_orig, training=True)
    plt.imshow(tf.cast(tf.clip_by_value(img_aug, 0, 255), tf.uint8))
    plt.title("augmenté"); plt.axis("off")

plt.tight_layout(); plt.show()
#----------------------------------------------------------------------------------
"""#Etape 2 - Préparation des Données"""
#----------------------------------------------------------------------------------
#import des modules nécessaires de Keras
from tensorflow import keras
from tensorflow.keras import layers

#Fonction qui construit un modèle CNN "baseline"
def build_baseline(num_classes):
    # Entrée du modèle (taille IMG_SIZE + 3 canaux couleur)
    inputs = keras.Input(shape=IMG_SIZE + (3,))

    #Normalisation des pixels (0-255 → 0-1), appliquée aussi en inference
    x = layers.Rescaling(1./255, name="rescale_01")(inputs)

    #Data augmentation intégrée au modèle (aléatoire, seulement en entraînement)
    data_aug = keras.Sequential([
        layers.RandomFlip('horizontal'),   #retournement horizontal
        layers.RandomZoom(0.10),           #zoom aléatoire
        layers.RandomContrast(0.30),       #contraste aléatoire
    ], name="augmentation_in_model")

    x = data_aug(x)  #application de l’augmentation (format batch 4D)

    #Convolution + MaxPooling (CNN classique)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3, padding='same', activation='relu')(x); x = layers.MaxPooling2D()(x)

    #Global pooling + dropout pour régularisation
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    #Couche de sortie avec softmax (classification multi-classes)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    #Construction et compilation du modèle
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#On instancie le modèle baseline avec le nombre de classes du dataset
model = build_baseline(len(class_names))

#Affichage du résumé du modèle
model.summary()
#----------------------------------------------------------------------------------
"""#Etape 3 - Conception et Implémentation du Modèle CNN"""
#----------------------------------------------------------------------------------
#Callback pour stopper l'entraînement si la val_accuracy ne s'améliore plus
cb = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)]

#Entraînement du modèle sur train_ds avec validation sur val_ds
hist = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cb)

#Affichage des courbes d'accuracy (train vs validation)
import matplotlib.pyplot as plt
plt.figure(); plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy'])
plt.legend(['acc','val_acc']); plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.show()

#Affichage des courbes de loss (train vs validation)
plt.figure(); plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss'])
plt.legend(['loss','val_loss']); plt.xlabel('epoch'); plt.ylabel('loss'); plt.show()
#----------------------------------------------------------------------------------
"""#Etape 4 - Entraînement du Modèle et Évaluation"""
#----------------------------------------------------------------------------------
#On importe numpy et les métriques de sklearn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

#On calcule les prédictions sur le jeu de test
y_true, y_pred = [], []
for xb, yb in test_ds:
    probs = model.predict(xb, verbose=0)         #prédictions (probabilités)
    y_pred.extend(np.argmax(probs, axis=1))      #indice de la classe prédite
    y_true.extend(np.argmax(yb.numpy(), axis=1)) #indice de la classe réelle

#Rapport de classification (précision, rappel, f1-score, support)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

#Matrice de confusion
cm = confusion_matrix(y_true, y_pred)

#Affichage graphique de la matrice de confusion
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest')          #affichage sous forme d'image
plt.title('Confusion Matrix'); plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)  #classes en colonnes
plt.yticks(range(len(class_names)), class_names)               #classes en lignes
plt.xlabel('Pred'); plt.ylabel('True')
plt.tight_layout(); plt.show()
#----------------------------------------------------------------------------------
"""#Etape 5 -  Évaluation du Modèle"""
#----------------------------------------------------------------------------------
#Hyperparamètres pour l'entraînement
LR_HEAD   = 1e-3        #learning rate pour la tête (phase 1)
LR_FT     = 1e-4        #learning rate pour le fine-tuning (phase 2)
DROPOUT   = 0.30        #taux de dropout pour régularisation
UNFREEZE_LAST = 30      #nombre de couches à "dégeler" pour le fine-tuning
EPOCHS1   = 6           #nb d'époques pour la phase 1
EPOCHS2   = 4           #nb d'époques pour la phase 2
PATIENCE  = 2           #patience pour early stopping
DO_ABLATION = True       #activer ou non l'ablation study

#On ajuste la taille des batchs si besoin
NEW_BATCH = BATCH
if NEW_BATCH != BATCH:
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.unbatch().batch(NEW_BATCH).prefetch(AUTOTUNE)
    val_ds   = val_ds.unbatch().batch(NEW_BATCH).prefetch(AUTOTUNE)
    test_ds  = test_ds.unbatch().batch(NEW_BATCH).prefetch(AUTOTUNE)

#Définition de plusieurs stratégies d’augmentation de données
aug_baseline = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="aug_baseline")

aug_plus_contrast = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="aug_plus_contrast")

aug_light = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.03),
], name="aug_light")

#Prétraitement pour MobileNetV2 : remet les pixels dans [-1,1]
to_m11 = layers.Rescaling(1./127.5, offset=-1)

#Fonction pour récupérer MobileNetV2 pré-entraîné sur ImageNet
def get_mobilenet_base():
    try:
        return tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet'
        )
    except Exception as e:
        from pathlib import Path
        cands = list(Path("/kaggle/input").rglob("mobilenet_v2_weights*_no_top.h5"))
        if cands:
            return tf.keras.applications.MobileNetV2(
                input_shape=IMG_SIZE + (3,), include_top=False, weights=str(cands[0])
            )
        print("Pas de poids ImageNet accessibles (internet OFF et aucun .h5 local). "
              "On passe à weights=None (moins performant).")
        return tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,), include_top=False, weights=None
        )

#Fonction qui construit le modèle de transfer learning
def build_tl_model(aug_layer):
    base = get_mobilenet_base()
    base.trainable = False #Phase 1 : on fige la base (feature extractor)
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = to_m11(inputs)
    x = aug_layer(x) #injection de la data augmentation choisie
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base

#Petit run rapide pour comparer les stratégies d’augmentation
def quick_val(aug_layer, epochs=2):
    m, _ = build_tl_model(aug_layer)
    h = m.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1,
                                                       restore_best_weights=True)])
    return float(h.history['val_accuracy'][-1])

#Ablation study:on compare baseline / contrast / light
if DO_ABLATION:
    scores = {
        "baseline":      quick_val(aug_baseline, epochs=2),
        "plus_contrast": quick_val(aug_plus_contrast, epochs=2),
        "light":         quick_val(aug_light, epochs=2),
    }
    print("Ablation (val_acc @2epochs):", scores)
    best_aug_name = max(scores, key=scores.get)
    aug_final = {"baseline": aug_baseline,
                 "plus_contrast": aug_plus_contrast,
                 "light": aug_light}[best_aug_name]
    print("→ On retient l’augmentation :", best_aug_name)
else:
    aug_final = aug_plus_contrast #valeur par défaut raisonnable

#TRANSFER LEARNING
#Phase 1 : entraînement de la tête uniquement
tl_model, base = build_tl_model(aug_final)
cb = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1),
]
hist1 = tl_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1, callbacks=cb, verbose=1)

#Phase 2 : fine-tuning partiel de la base
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:   #on garde figées toutes sauf les dernières couches
    layer.trainable = False
tl_model.compile(optimizer=keras.optimizers.Adam(LR_FT),
                 loss='categorical_crossentropy', metrics=['accuracy'])
hist2 = tl_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2, callbacks=cb, verbose=1)

#Résumé des meilleurs scores de validation
def best_val(h):
    i = int(np.argmax(h.history['val_accuracy']))
    return {"best_epoch": i+1,
            "val_acc": float(h.history['val_accuracy'][i]),
            "val_loss": float(h.history['val_loss'][i])}
try:
    print("Phase1 best:", best_val(hist1))
    print("Phase2 best:", best_val(hist2))
except Exception:
    pass
#----------------------------------------------------------------------------------
"""#Etape 6 - Améliorations et Expérimentations"""
#----------------------------------------------------------------------------------
#On importe les librairies nécessaires
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt

#0)On récupère le "meilleur modèle" disponible (soit tl_model, soit model)
model_to_explain = globals().get('tl_model', globals().get('model'))
assert model_to_explain is not None, "Aucun modèle chargé (tl_model ou model)."
input_hw = model_to_explain.input_shape[1:3]

#SALIENCY MAP
def saliency_on_image(img_4d):
    """
    Fonction qui calcule une "saliency map".
    → Montre quels pixels de l'image influencent le plus la prédiction du modèle.
    img_4d: (1,H,W,3) en float32 (valeurs entre 0 et 255).
    Retourne: heatmap HxW normalisée, indice de la classe prédite, et confiance associée.
    """
    img = tf.cast(img_4d, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img) #on surveille les pixels
        preds = model_to_explain(img, training=False) #prédiction
        idx = tf.argmax(preds[0]) #classe prédite
        score = preds[:, idx]     #score de cette classe

    #Gradient du score par rapport à l’image (sensibilité pixel par pixel)
    grads = tape.gradient(score, img)[0]
    sal = tf.math.reduce_max(tf.abs(grads), axis=-1)       #intensité max par pixel
    sal = sal / (tf.reduce_max(sal) + 1e-8)                #normalisation entre 0 et 1
    return sal.numpy(), int(idx.numpy()), float(preds[0, idx].numpy())

def overlay_heatmap(img_u8, heat):
    """Superpose la heatmap colorée ('jet') à l’image originale."""
    cmap = plt.get_cmap('jet')
    hm_rgb = (cmap(heat)[..., :3] * 255).astype("uint8")   #heatmap en couleur
    over = (0.6 * img_u8 + 0.4 * hm_rgb).clip(0, 255).astype("uint8")  #fusion
    return hm_rgb, over

#Démo sur une image du test set
xb_demo, yb_demo = next(iter(test_ds))
x1 = xb_demo[0:1]                                           #image unique
true_idx = int(tf.argmax(yb_demo[0]).numpy())               #vraie étiquette

sal, pred_idx, conf = saliency_on_image(x1)
hm_rgb, overlay = overlay_heatmap(xb_demo[0].numpy().astype("uint8"), sal)

#Affichage : image originale, saliency map, overlay
plt.figure(figsize=(11,4))
plt.subplot(1,3,1); plt.imshow(xb_demo[0].numpy().astype("uint8"))
plt.title(f"Original — True: {class_names[true_idx]}"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(sal, cmap='jet'); plt.title("Saliency (importance)"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(overlay)
plt.title(f"Pred: {class_names[pred_idx]} (conf {conf:.2f})"); plt.axis('off')
plt.tight_layout(); plt.show()
print("💡 Interprétation : zones rouges/jaunes = pixels auxquels la prédiction est la plus sensible.")

#OCCLUSION SENSITIVITY
def occlusion_map(img_u8, patch=20, stride=20, baseline=None):
    """
    Technique d’occlusion : on cache des parties de l'image pour voir où la confiance chute.
    → Plus la chute est forte, plus la zone était importante.
    Renvoie une carte HxW (valeurs normalisées entre 0 et 1).
    """
    H, W, _ = img_u8.shape
    img = img_u8.astype(np.float32)
    p0 = model_to_explain(img[None, ...], training=False).numpy()[0]
    cls = int(np.argmax(p0))               #classe prédite
    base_score = float(p0[cls])            #confiance de base

    #si baseline non donnée, on "occlut" avec un gris moyen
    if baseline is None:
        baseline = np.array([127.5, 127.5, 127.5], dtype=np.float32)

    heat = np.zeros((H, W), dtype=np.float32)
    #On balaye l’image par patchs (fenêtres carrées)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2, x2 = min(y+patch, H), min(x+patch, W)
            tmp = img.copy()
            tmp[y:y2, x:x2, :] = baseline                 #on cache la zone
            p = model_to_explain(tmp[None, ...], training=False).numpy()[0][cls]
            drop = max(0.0, base_score - float(p))        #perte de confiance
            heat[y:y2, x:x2] = drop
    if heat.max() > 0:
        heat /= heat.max()
    return heat, cls, base_score

#Application de l’occlusion sensitivity sur une image
occ, cls_idx, base_conf = occlusion_map(xb_demo[0].numpy().astype("uint8"), patch=24, stride=16)
hm_occ, over_occ = overlay_heatmap(xb_demo[0].numpy().astype("uint8"), occ)

#Affichage : image originale, heatmap d’occlusion, overlay
plt.figure(figsize=(11,4))
plt.subplot(1,3,1); plt.imshow(xb_demo[0].numpy().astype("uint8")); plt.title("Image"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(occ, cmap='jet'); plt.title("Occlusion (importance)"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(over_occ); plt.title(f"Classe analysée: {class_names[cls_idx]} (conf {base_conf:.2f})"); plt.axis('off')
plt.tight_layout(); plt.show()
print("💡 Occlusion : plus c'est chaud, plus masquer la zone fait chuter la confiance.")

#ANALYSE DES ERREURS (confusions)
from sklearn.metrics import confusion_matrix

#On récupère les prédictions sur tout le test set
y_true, y_pred = [], []
for xb, yb in test_ds:
    preds = model_to_explain.predict(xb, verbose=0)
    y_true.extend(np.argmax(yb.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
y_true = np.array(y_true); y_pred = np.array(y_pred)

#Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)  #on ignore la diagonale
pairs = [(i, j, cm_off[i, j]) for i in range(len(class_names)) for j in range(len(class_names)) if i != j]
pairs.sort(key=lambda t: t[2], reverse=True)
top_pairs = [p for p in pairs if p[2] > 0][:2]  #on garde les 2 pires confusions

print("\nTop confusions (True → Pred | count) :")
for i, j, c in top_pairs:
    print(f"  {class_names[i]} → {class_names[j]} | {c}")

#isualisation des exemples pour la pire confusion
if top_pairs:
    ti, pj, _ = top_pairs[0]
    shown = 0
    plt.figure(figsize=(12,6))
    for xb, yb in test_ds:
        preds = model_to_explain.predict(xb, verbose=0)
        for k in range(xb.shape[0]):
            if np.argmax(yb[k].numpy()) == ti and np.argmax(preds[k]) == pj:
                sal_k, _, conf_k = saliency_on_image(xb[k:k+1])
                _, overlay_k = overlay_heatmap(xb[k].numpy().astype("uint8"), sal_k)
                plt.subplot(2,2,shown+1); plt.imshow(overlay_k); plt.axis('off')
                plt.title(f"True:{class_names[ti]} → Pred:{class_names[pj]} (conf {conf_k:.2f})")
                shown += 1
                if shown == 2: break
        if shown == 2: break
    plt.suptitle(f"Saliency — Pire confusion : {class_names[ti]} → {class_names[pj]}")
    plt.tight_layout(); plt.show()

#PISTES D’AMÉLIORATION
def suggestions_for_pair(true_name, pred_name):
    tips = [
        "- Augmentations ciblées :",
        "  Brightness/Contrast si l’éclairage varie (ex: 'sea' ↔ 'glacier').",
        "  RandomTranslation/RandomZoom si le cadrage change ('buildings' ↔ 'street').",
        "  Légère ColorJitter (si dispo) pour distinguer teintes (ciel/mer/neige).",
        "Fine-tuning : ouvrir +10 couches (UNFREEZE_LAST) et baisser LR_FT (ex: 5e-5).",
        "Si textures proches ('forest' ↔ 'mountain') : rotation ±0.08 + contrast 0.15.",
    ]
    print(f"\nPistes pour réduire {true_name} → {pred_name} :")
    print("\n".join(tips))

#On génère des suggestions pour la pire confusion
for i, j, _ in top_pairs[:1]:
    suggestions_for_pair(class_names[i], class_names[j])
#----------------------------------------------------------------------------------
"""#Etape 7 -  Interprétation et Visualisation des Résultats"""
#----------------------------------------------------------------------------------
#On importe toutes les librairies nécessaires : numpy, tensorflow, matplotlib, opencv + les couches Keras
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, cv2
from tensorflow.keras import layers

#On choisit le modèle à expliquer : si 'tl_model' existe on le prend sinon on prend 'model'
model_to_explain = tl_model if 'tl_model' in globals() else model  #prends le meilleur

#On cherche la dernière couche convolutive du modèle car Grad-CAM se base sur une couche conv
last_conv = None
for layer in reversed(model_to_explain.layers):
    if isinstance(layer, layers.Conv2D):
        last_conv = layer.name; break
print("Dernière conv:", last_conv)

#Fonction Grad-CAM : génère une heatmap qui montre les zones de l’image les plus importantes pour la prédiction
def gradcam(img_tensor, model, last_conv_layer_name):
    # On crée un "sous-modèle" qui sort à la fois la dernière feature map et la prédiction
    conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    #On enregistre les calculs pour la rétropropagation des gradients
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)   #sortie de la couche conv + prédiction finale
        idx = tf.argmax(preds[0])                 #on prend la classe prédite (indice max)
        class_score = preds[:, idx]               #score de la classe choisie

    #Calcul du gradient du score de la classe par rapport à la sortie de la couche conv
    grads = tape.gradient(class_score, conv_out)

    #Moyenne des gradients sur les dimensions spatiales pondérations pour chaque canal de la feature map
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    #On récupère la feature map correspondante pour une seule image
    conv_out = conv_out[0]

    #Produit scalaire entre la feature map et les poids des gradients => importance par canal
    heat = conv_out @ pooled[..., tf.newaxis]
    heat = tf.squeeze(heat)

    #normalisation : on garde que les activations positives et on met entre 0 et 1
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy(), int(idx)

#On prend une image du dataset de test (la première du batch)
for xb, yb in test_ds.take(1):
    img0 = xb[0:1]                                 #une seule image
    true_idx = int(tf.argmax(yb[0]).numpy())       #vraie classe
    break

#On applique Grad-CAM sur l'image choisie
hm, pred_idx = gradcam(img0, model_to_explain, last_conv)

#On redimensionne la heatmap à la taille d'entrée de l'image
hm = tf.image.resize(hm[...,None], IMG_SIZE).numpy().squeeze()

#Conversion de l’image et préparation du rendu avec la heatmap
img_disp = img0[0].numpy().astype("uint8")
heatmap = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)   #heatmap en couleur
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)                #passage en RGB
overlay  = (0.6*img_disp + 0.4*heatmap).clip(0,255).astype("uint8") #superposition image + heatmap

#Affichage des résultats : image originale, heatmap Grad-CAM, overlay
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(img_disp); plt.title(f"True: {class_names[true_idx]}"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(hm, cmap='jet'); plt.title("Grad-CAM"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(overlay); plt.title(f"Pred: {class_names[pred_idx]}"); plt.axis('off')
plt.tight_layout(); plt.show()