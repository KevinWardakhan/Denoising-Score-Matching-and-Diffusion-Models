## Que regarder dans notre code :

- **`eval.ipynb`** *(notebook principal)*  
  Notebook “pré-compilé” qui charge un run (CelebA) et montre les expériences de sampling **VE-SDE** : PC sampler, trajectoires (snapshots), comparaison **PC vs P-only vs C-only**, et une étude **C-only vs SNR**.

- **`LD_vs_ALD_GMM.ipynb`** *(toy experiment)*  
  Expérience 2D sur une GMM avec **score exact** : on compare **Langevin single-scale** vs **Annealed Langevin Dynamics (ALD)**, avec visualisations de trajectoires et histogrammes de proportions (illustration du collapse multi-mode et de l’intérêt de ALD).

- **`models.py`**  
  Définit l’architecture du réseau (U-Net + embedding de \(\sigma\)) utilisée pour le score model.
---

## Notes d’exécution

- À part **`LD_vs_ALD_GMM.ipynb`**, les notebooks **ne sont pas directement re-runnables** sans les **poids entraînés** et le dataset **CelebA** (paths / stockage non fournis ici). Les notebooks sont fournis surtout pour montrer clairement le code et les résultats obtenus.

---

## Moins utile à lire (sauf si vous cherchez un détail précis)

- **`train_ema.py`**  
  Le script d’entraînement complet. Il n’est pas très utile pour lire le projet en première passe, sauf si vous voulez voir précisément **comment la loss VE/DSM est implémentée**, ou la logique de logging / EMA / sauvegardes. Le script peut être dur à lire mais la Loss est définie au tout début.

- **`train_cifar.py`**  
  Le script d’entraînement complet sur cifar10.