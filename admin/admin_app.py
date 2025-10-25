# admin/admin_app.py
import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
ADMIN_KEY = os.getenv("ADMIN_KEY", "change_this_admin_key")

st.set_page_config(page_title="SmartBizBot Admin", layout="wide")
st.title("SmartBizBot — Admin Console")

st.sidebar.header("Actions")
act = st.sidebar.selectbox("Action", ["Create Tenant","Upload CSV","Add Q/A","Test Chat","List Tenants"])

headers = {"x-admin-key": ADMIN_KEY}

if act == "Create Tenant":
    tid = st.text_input("Tenant ID (unique key)")
    name = st.text_input("Business name")
    if st.button("Create Tenant"):
        resp = requests.post(f"{API_BASE}/tenant/create", json={"tenant_id": tid, "business_name": name}, headers=headers)
        st.write(resp.json())

if act == "Upload CSV":
    tid = st.text_input("Tenant ID")
    uploaded = st.file_uploader("Upload CSV (question,answer)", type=["csv"])
    if uploaded and st.button("Upload to Tenant"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        resp = requests.post(f"{API_BASE}/tenant/{tid}/upload_csv", files=files, headers=headers)
        st.write(resp.json())

if act == "Add Q/A":
    tid = st.text_input("Tenant ID")
    q = st.text_input("Question")
    a = st.text_area("Answer")
    if st.button("Add Q/A"):
        resp = requests.post(f"{API_BASE}/tenant/{tid}/add_qa", data={"question": q, "answer": a}, headers=headers)
        st.write(resp.json())

if act == "Test Chat":
    tid = st.text_input("Tenant ID")
    q = st.text_input("Ask question")
    if st.button("Ask"):
        resp = requests.post(f"{API_BASE}/chat", json={"tenant_id": tid, "question": q})
        st.write(resp.json())

if act == "List Tenants":
    resp = requests.get(f"{API_BASE}/tenants", headers=headers)
    st.write(resp.json())
