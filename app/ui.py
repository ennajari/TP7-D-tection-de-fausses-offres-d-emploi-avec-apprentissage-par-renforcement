"""
Interface utilisateur pour la détection de fausses offres d'emploi
TP7 - Apprentissage par Renforcement appliqué à la classification textuelle
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
import base64

# Définition de la structure du modèle DQN (doit correspondre à celle utilisée pendant l'entraînement)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Fonction de prétraitement du texte (identique à celle utilisée pendant l'entraînement)
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)  # Suppression des balises HTML
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)  # Suppression des caractères spéciaux
    text = re.sub(r'\s+', ' ', text)  # Normalisation des espaces
    return text.lower().strip()

# Fonction pour évaluer une offre d'emploi
def evaluate_job(title, description, model, vectorizer):
    # Préparation du texte
    text = clean_text(title + ' ' + description)
    
    # Vectorisation
    features = vectorizer.transform([text]).toarray().astype(np.float32)
    
    # Prédiction
    with torch.no_grad():
        state_tensor = torch.FloatTensor(features[0]).unsqueeze(0)
        q_values = model(state_tensor)
        prediction = torch.argmax(q_values[0]).item()
        proba_fake = F.softmax(q_values, dim=1)[0][1].item()
    
    return prediction, q_values[0].numpy(), proba_fake

# Fonction pour générer un graphique des q-values
def plot_q_values(q_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = ['Authentique', 'Frauduleuse']
    colors = ['#2ecc71', '#e74c3c'] if q_values[0] > q_values[1] else ['#e74c3c', '#2ecc71']
    
    ax.bar(classes, q_values, color=colors)
    ax.set_title('Q-values pour cette offre d\'emploi', fontsize=15)
    ax.set_ylabel('Q-value', fontsize=12)
    
    # Annotation des valeurs
    for i, v in enumerate(q_values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)
    
    # Ajout d'une ligne horizontale pour aider à visualiser la différence
    max_value = max(q_values)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Calcul de la confiance basée sur la différence des Q-values
    confidence = abs(q_values[0] - q_values[1])
    ax.text(0.5, max_value * 1.1, f'Différence: {confidence:.4f}', 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    return fig

# Fonction pour générer un radar chart des caractéristiques de risque
def plot_risk_features(title, description):
    # Caractéristiques de risque à évaluer
    risk_features = {
        'Urgence': len(re.findall(r'urgent|immediate|asap|quick', title.lower() + ' ' + description.lower())),
        'Argent facile': len(re.findall(r'easy money|rich|wealth|income|profit|earn \$', description.lower())),
        'Informations personnelles': len(re.findall(r'bank details|personal info|credit card|ssn|passport', description.lower())),
        'Travail à domicile': len(re.findall(r'work from home|remote|wfh', title.lower() + ' ' + description.lower())),
        'Promesses exagérées': len(re.findall(r'guarantee|promise|best opportunity|lifetime', description.lower()))
    }
    
    # Normalisation des scores (entre 0 et 5)
    max_val = max(risk_features.values()) if max(risk_features.values()) > 0 else 1
    risk_features = {k: min(5, v * (5/max_val)) for k, v in risk_features.items()}
    
    # Création du radar chart
    labels = list(risk_features.keys())
    values = list(risk_features.values())
    
    # Ajout du premier point à la fin pour fermer le polygone
    values.append(values[0])
    labels.append(labels[0])
    
    # Angles pour chaque axe
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    ax.plot(angles, values, color='#3498db', linewidth=2)
    
    # Étiquettes et graduation
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    
    # Ajout des valeurs sur chaque axe
    for angle, value, label in zip(angles[:-1], values[:-1], labels[:-1]):
        ax.text(angle, value + 0.3, f"{value:.1f}", ha='center', va='center')
    
    plt.title('Analyse des caractéristiques à risque', size=15)
    
    return fig

# Interface utilisateur avec Streamlit
def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Détecteur d'offres d'emploi frauduleuses",
        page_icon="🕵️",
        layout="wide"
    )
    
    # En-tête
    st.title("🕵️ Détecteur d'offres d'emploi frauduleuses")
    st.markdown("""
    Cette application utilise l'apprentissage par renforcement pour détecter les offres d'emploi frauduleuses. 
    Entrez le titre et la description d'une offre pour l'analyser.
    """)
    
    # Chargement du modèle et du vectoriseur
    @st.cache_resource
    def load_model():
        try:
            # Chargement du vectoriseur
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Chargement du modèle
            input_dim = 5000  # Doit correspondre à max_features du vectoriseur
            model = DQN(input_dim, 2)
            model.load_state_dict(torch.load('fake_job_dqn_model.pth', map_location=torch.device('cpu')))
            model.eval()
            
            return model, vectorizer
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            return None, None
    
    model, vectorizer = load_model()
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["Analyse d'offre", "Exemples prédéfinis", "À propos"])
    
    # Onglet 1: Analyse d'offre
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entrez les détails de l'offre d'emploi")
            job_title = st.text_input("Titre de l'offre", placeholder="Ex: Data Scientist - Remote Work")
            job_description = st.text_area("Description de l'offre", 
                                          height=200, 
                                          placeholder="Entrez la description complète de l'offre...")
            
            analyze_button = st.button("Analyser l'offre", type="primary")
            
        with col2:
            st.subheader("Résultat de l'analyse")
            
            if analyze_button and job_title and job_description:
                with st.spinner("Analyse en cours..."):
                    # Évaluation de l'offre
                    prediction, q_values, proba_fake = evaluate_job(job_title, job_description, model, vectorizer)
                    
                    # Affichage du résultat avec une jauge
                    if prediction == 1:
                        st.error("⚠️ ATTENTION: Cette offre est potentiellement FRAUDULEUSE!")
                        risk_level = proba_fake
                    else:
                        st.success("✅ Cette offre semble AUTHENTIQUE.")
                        risk_level = proba_fake
                    
                    # Jauge de risque
                    st.markdown("### Niveau de risque")
                    st.progress(risk_level)
                    st.write(f"Probabilité de fraude: {risk_level:.2%}")
                    
                    # Graphiques
                    st.pyplot(plot_q_values(q_values))
                    
                    # Analyse des caractéristiques de risque
                    st.markdown("### Analyse des facteurs de risque")
                    st.pyplot(plot_risk_features(job_title, job_description))
                    
                    # Conseils en fonction du résultat
                    st.markdown("### Recommandations")
                    if prediction == 1:
                        st.warning("""
                        - Méfiez-vous des demandes d'informations personnelles ou financières
                        - Recherchez l'entreprise en ligne avant de postuler
                        - Vérifiez si l'entreprise a un site web professionnel et des avis
                        - Ne payez jamais de frais pour postuler à un emploi
                        """)
                    else:
                        st.info("""
                        - Cette offre semble légitime, mais restez toujours vigilant
                        - Vérifiez l'entreprise sur des sites d'avis comme Glassdoor
                        - Ne communiquez jamais d'informations sensibles avant de confirmer la légitimité de l'offre
                        """)
            else:
                st.info("Entrez le titre et la description d'une offre d'emploi, puis cliquez sur 'Analyser l'offre'.")
    
    # Onglet 2: Exemples prédéfinis
    with tab2:
        st.subheader("Exemples d'offres d'emploi")
        st.write("Cliquez sur un exemple pour l'analyser instantanément.")
        
        examples = [
            {
                "title": "Software Developer - Junior Position",
                "description": "We're looking for a passionate junior developer to join our team. Requirements: Basic knowledge of programming languages like Python or JavaScript. We offer mentorship, competitive salary, and a friendly work environment. Send your resume to our official HR email.",
                "type": "Authentique"
            },
            {
                "title": "Marketing Specialist - Work from Home",
                "description": "Marketing position with flexible hours. Help promote our products online. Experience with social media marketing preferred. We offer training and competitive commission. Contact our HR department through our official website.",
                "type": "Authentique"
            },
            {
                "title": "URGENT - Make Money Fast - Work From Home!!!",
                "description": "Make $5000 weekly working just 2 hours a day! No experience needed! Just send us your bank details to get started immediately. Limited positions available! Act now before it's too late! 100% guaranteed income!",
                "type": "Frauduleuse"
            },
            {
                "title": "Personal Assistant Needed - Immediate Start",
                "description": "Looking for a reliable personal assistant. Easy work, high pay! $500 per day just for checking emails. No experience required. Must be able to receive payments to your personal bank account. Contact us at quickmoney@gmail.com right away!",
                "type": "Frauduleuse"
            }
        ]
        
        for i, example in enumerate(examples):
            col1, col2, col3 = st.columns([3, 6, 1])
            with col1:
                st.write(f"**{example['title']}**")
            with col2:
                st.write(example['description'][:100] + "...")
            with col3:
                analyze_example = st.button(f"Analyser", key=f"example_{i}")
            
            if analyze_example:
                with st.spinner("Analyse en cours..."):
                    # Évaluation de l'offre
                    prediction, q_values, proba_fake = evaluate_job(example['title'], example['description'], model, vectorizer)
                    
                    # Affichage détaillé de l'exemple
                    st.markdown("---")
                    st.subheader(f"Analyse de l'exemple: {example['title']}")
                    st.write(f"**Description complète:** {example['description']}")
                    
                    # Affichage du résultat
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ FRAUDULEUSE")
                        else:
                            st.success("✅ AUTHENTIQUE")
                        
                        st.markdown(f"**Étiquette réelle:** {example['type']}")
                        st.markdown(f"**Probabilité de fraude:** {proba_fake:.2%}")
                    
                    with col2:
                        st.pyplot(plot_q_values(q_values))
                    
                    st.pyplot(plot_risk_features(example['title'], example['description']))
                    st.markdown("---")
    
    # Onglet 3: À propos
    with tab3:
        st.subheader("À propos de cette application")
        
        st.markdown("""
        ### Contexte du projet
        
        Cette application a été développée dans le cadre du TP7 sur l'utilisation de l'apprentissage par renforcement 
        pour des tâches de classification de texte. L'objectif était de modéliser un problème réel comme un environnement 
        d'apprentissage par renforcement et d'entraîner un agent à prendre des décisions intelligentes.
        
        ### Méthodologie
        
        1. **Préparation des données**: Les offres d'emploi ont été vectorisées en utilisant TF-IDF.
        2. **Création d'un environnement personnalisé**: Chaque état représente une offre d'emploi vectorisée.
        3. **Apprentissage par renforcement**: Un agent DQN (Deep Q-Network) a été entraîné à distinguer les offres authentiques des frauduleuses.
        4. **Interface utilisateur**: Cette application permet d'utiliser le modèle entraîné pour analyser de nouvelles offres.
        
        ### Technologies utilisées
        
        - **Apprentissage par renforcement**: Deep Q-Network (DQN)
        - **Traitement du texte**: TF-IDF Vectorization
        - **Framework ML**: PyTorch
        - **Interface utilisateur**: Streamlit
        
        ### Fonctionnalités supplémentaires
        
        - Analyse des caractéristiques à risque dans les offres
        - Visualisation des Q-values
        - Exemples prédéfinis pour démonstration
        - Recommandations personnalisées
        """)
        
        st.info("""
        **Note**: Cette application est une démonstration éducative. 
        Dans un contexte réel, des modèles plus complexes et davantage de données seraient nécessaires 
        pour une détection fiable des offres frauduleuses.
        """)
        
        # Caractéristiques de détection
        st.subheader("Caractéristiques courantes des offres frauduleuses")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - Promesses de revenus élevés pour peu d'effort
            - Demandes d'informations personnelles ou bancaires
            - Absence de prérequis professionnels
            - Erreurs grammaticales et fautes d'orthographe
            - Utilisation excessive de majuscules et de points d'exclamation
            """)
            
        with col2:
            st.markdown("""
            - Adresses email non professionnelles (Gmail, Hotmail, etc.)
            - Sentiment d'urgence ("Postulez maintenant!", "Opportunité limitée!")
            - Manque de détails sur l'entreprise ou les responsabilités
            - Offres trop belles pour être vraies
            - Demande de paiement pour postuler ou pour formation
            """)

# Lancement de l'application
if __name__ == "__main__":
    main()