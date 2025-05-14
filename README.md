This code implements a Quantum Diffusion Model (QDM) based on the paper by Cacioppo et al. for the Quantum Diffusion Models course exam.


The model uses parameterized quantum circuits (PQCs) in place of traditional artificial neural networks to generate quantum states. The implementation presents a hybrid approach, with a classical forward process and a quantum backward process, using the Reverse-Bottleneck architecture. A latent version of the model has been created, with conditioning capabilities. The model was implemented using PennyLane and tested on the MNIST dataset.

https://arxiv.org/abs/2311.15444 

The quantum diffusion model implementation includes:

\begin{itemize}
    \item \textbf{Forward Process}: Un autoencoder classico con rumore gaussiano complesso
    \item \textbf{Backward Process}: Un circuito quantistico parametrizzato (PQC) utilizzando PennyLane
    \item \textbf{Reverse-Bottleneck Architecture}: Un design di circuito quantistico a tre blocchi con qubit di dati e qubit ancillari
    \item \textbf{Latent Model}: Riduzione della dimensionalità con un autoencoder classico seguito da modellazione di diffusione quantistica
    \item \textbf{Conditioning Support}: Capacità di generare campioni condizionati per classe
    \item \textbf{Evaluation Metrics}: Include FID, distanza di Wasserstein e ROC-AUC
    \item \textbf{MNIST Processing}: Funzionalità per pre-elaborare e utilizzare i dati MNIST
\end{itemize}

The code structure is organized into logical sections that follow the paper's methodology:

\begin{itemize}
    \item Preparazione dei dati di input
    \item Processo di diffusione forward (classico)
    \item Processo di diffusione backward (quantistico)
    \item Procedure di addestramento e campionamento
    \item Valutazione
    \item Visualizzazione
\end{itemize}



