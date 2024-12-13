import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Détection de SPAM dans les Commentaires YouTube",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


navbar_css = """
<style>
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    background-color: #f8f9fa;
    border-bottom: 1px solid #e0e0e0;
}
.navbar .nav-links {
    display: flex;
    gap: 20px;
}
.navbar a {
    text-decoration: none;
    color: #007BFF;
    font-weight: bold;
    font-size: 16px;
}
.navbar a:hover {
    color: #0056b3;
}
</style>
"""

st.markdown(navbar_css, unsafe_allow_html=True)


navbar_html = """
<div class="navbar">
    <div class="logo">
        <a href="#home" style="font-size: 20px; color: #333;">🛡️ Détection SPAM</a>
    </div>
    <div class="nav-links">
        <a href="#home">Accueil</a>
        <a href="#test">Tester un commentaire</a>
        <a href="#details">Détails du modèle</a>
        <a href="#data">Données</a>
    </div>
</div>
"""

st.markdown(navbar_html, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = joblib.load('spam_detector_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Modèle introuvable. Assurez-vous que 'spam_detector_model.pkl' est présent.")
        return None


@st.cache_resource
def load_data():
    data = pd.concat([
        pd.read_csv('./content/Youtube01-Psy.csv'),
        pd.read_csv('./content/Youtube02-KatyPerry.csv'),
        pd.read_csv('./content/Youtube03-LMFAO.csv'),
        pd.read_csv('./content/Youtube04-Eminem.csv'),
        pd.read_csv('./content/Youtube05-Shakira.csv')
    ])
    seed = 1234
    x_train, x_test, y_train, y_test = train_test_split(
        data['CONTENT'], data['CLASS'], test_size=0.2, random_state=seed
    )
    return x_train, x_test, y_train, y_test


model = load_model()
x_train, x_test, y_train, y_test = load_data()


st.title("🛡️ Détection de SPAM dans les Commentaires YouTube")


st.markdown('<a id="test"></a>', unsafe_allow_html=True)
st.subheader("Tester un commentaire")
comment = st.text_area("Entrez un commentaire pour tester :", "")

if st.button("Prédire"):
    if model and comment.strip():
        prediction = model.predict([comment])[0]
        if prediction == 1:
            st.error("🚨 Ce commentaire est un SPAM.")
        else:
            st.success("✅ Ce commentaire n'est pas un SPAM.")
    else:
        st.warning("Veuillez entrer un commentaire valide.")


st.markdown('<a id="details"></a>', unsafe_allow_html=True)
st.subheader("Détails du modèle")
if model:
    st.markdown("### Performances :")
    st.markdown(f"- Score moyen : {model.score(x_test, y_test):.2f}")


st.markdown('<a id="data"></a>', unsafe_allow_html=True)
if st.checkbox("Afficher les données d'entraînement"):
    st.subheader("Données d'entraînement")
    data = pd.concat([
        pd.read_csv('./content/Youtube01-Psy.csv'),
        pd.read_csv('./content/Youtube02-KatyPerry.csv'),
        pd.read_csv('./content/Youtube03-LMFAO.csv'),
        pd.read_csv('./content/Youtube04-Eminem.csv'),
        pd.read_csv('./content/Youtube05-Shakira.csv')
    ])
    st.dataframe(data)
