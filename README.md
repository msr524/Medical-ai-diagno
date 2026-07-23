# MedVision AI: Production-Grade Medical Diagnostic Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*An enterprise-grade, end-to-end computer vision and explainable AI platform designed for automated chest X-ray classification, clinical interpretability via Grad-CAM, automated LLM medical reporting, and robust database audit logging.*

</div>

---

## 📌 Executive Summary

**MedVision AI** bridges the gap between deep learning research and clinical deployment. Built with a decoupled architecture, it pairs a high-performance **FastAPI** backend with a dynamic **Streamlit** multi-view interface. The core vision pipeline utilizes a fine-tuned Hugging Face ResNet architecture with mathematically synchronized preprocessing and dynamic class label mapping to eliminate common inference failure modes. For clinical trust, the system features a custom **Grad-CAM (Gradient-weighted Class Activation Mapping)** engine to visualize pathological focus areas, accompanied by an intelligent **Gemini API** reporting service equipped with fallback handlers for rate limits.

---

## 🏗️ System Architecture & Workflow

```text
┌─────────────────┐       HTTP Multipart POST       ┌──────────────────────┐
│  Streamlit UI   │ ──────────────────────────────> │   FastAPI Backend    │
│    (Frontend)   │ <────────────────────────────── │  (Stateless REST API)│
└─────────────────┘      JSON Payload & Image     └──────────┬───────────┘
         │                                                    │
         │ (Toggles to View History)                          ▼
         │                                         ┌─────────────────────┐
         └───────────────────────────────────────> │ SQLite Audit Engine │
                                                   └─────────────────────┘
         │ 
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌──────────────────────────────┐                ┌─────────────────────────────┐
│    Hugging Face ResNet-18    │                │      Gemini LLM Engine      │
│   (AutoImageProcessor + XAI) │                │    (Draft Report & Fallback)│
└──────────────────────────────┘                └─────────────────────────────┘
