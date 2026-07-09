import streamlit as st
import hashlib
from datetime import datetime
import io

def get_user_id():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]
    return st.session_state.user_id

def create_csv_download(df_summary):
    output = io.StringIO()
    df_summary.to_csv(output, index=False)
    return output.getvalue()
