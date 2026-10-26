"""Animated author cards, embedded locally with no external image hosting."""
from pathlib import Path
import base64
import json

AUTHORS = [
    dict(name="Shivam Tomar", roll="25n0019", email="st717338@gmail.com", initials="ST", stem="shivam"),
    dict(name="Ankit Singh Yadav", roll="25n0015", email="mrankit955@gmail.com", initials="AY", stem="ankit"),
]

def build_html(root):
    authors=[]
    for entry in AUTHORS:
        author=dict(entry)
        author["photo"]=""
        for suffix,mime in [(".jpg","image/jpeg"),(".jpeg","image/jpeg"),(".png","image/png")]:
            path=Path(root)/"assets"/"authors"/(author["stem"]+suffix)
            if path.is_file():
                author["photo"]="data:"+mime+";base64,"+base64.b64encode(path.read_bytes()).decode("ascii")
                break
        authors.append(author)
    template=Path(__file__).with_name("authors.html").read_text(encoding="utf-8")
    return template.replace("__AUTHORS_JSON__",json.dumps(authors).replace("<",r"\u003c"))

def render(root):
    import streamlit as st
    import streamlit.components.v1 as components
    st.markdown("<style>.block-container{max-width:none!important;padding-left:1.5rem;padding-right:1.5rem}</style>",unsafe_allow_html=True)
    st.title("Authors")
    st.caption("Meet the people behind Seismic Velocity Modelling Lab. Select a photo to view their details.")
    components.html(build_html(root),height=900,scrolling=True)
