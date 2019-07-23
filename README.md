---

# Tesi magistrale "Model Agnostic solution of CSPs with Deep Learning"
#### Candidato: Francesco Giovanelli
#### Relatore: prof.ssa Michela Milano
#### Correlatori: dott. Michele Lombardi, dott. Andrea Galassi

---

## Descrizione script

Breve descrizione degli script creati:

+ **discriminator**: rete neurale Discriminatore, versione iniziale (funzione di attivazione ReLU) 
+ **discriminator_leaky**: rete neurale Discriminatore, seconda versione per GAN (funzione di attivazione Leaky ReLU)
+ **generator**: rete neurale Generatore, versione iniziale (funzione di attivazione ReLU) 
+ **generator_leaky**: rete neurale Generatore, seconda versione per GAN (funzione di attivazione Leaky ReLU)
+ **my_utils**: file con funzioni di utilità (caricamento dataset da file, creazione logger, funzioni di valutazione feasibility, ecc.)
+ **gan**: modello GAN
+ **wgan**: modello WGAN con Gradient Penalty
+ **dataset_generator_for_discriminator_network**: utility per produrre dataset con dati unfeasible, da usare per Discriminatore (e GAN, se necessario)
+ **SVM**: algoritmo SVM in versione Generatore e Discriminatore

---