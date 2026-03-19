import gradio as gr
import torch
import asyncio
import pandas as pd
from src.generator.dcgan_engine import Generator, Discriminator
from src.agents.critic_agent import CriticAgent
from config import cfg
from PIL import Image
import numpy as np

# System Core
device = "cpu"
gen = Generator().to(device)
disc = Discriminator().to(device)
agent = CriticAgent()

async def live_research_pipeline(seed, lr_val, batch_size):
    # 1. Neural Simulation with dynamic params
    torch.manual_seed(seed)
    noise = torch.randn(batch_size, cfg.LATENT_DIM).to(device)
    with torch.no_grad():
        fake_tensor = gen(noise).cpu()
    
    # Display the first image of the batch
    img_array = (fake_tensor[0].permute(1, 2, 0).numpy() * 0.5) + 0.5
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # 2. Simulated Live Analytics
    epochs = list(range(1, 11))
    g_loss = [0.9 + np.random.normal(0, 0.05) for _ in epochs]
    d_loss = [0.3 + np.random.normal(0, 0.02) for _ in epochs]
    chart_data = pd.DataFrame({"Epoch": epochs, "Gen Loss": g_loss, "Disc Loss": d_loss})
    
    # 3. Cognitive Agent Expert Audit
    report = await agent.audit_generation(epoch="LIVE-RESEARCH", loss_g=g_loss[-1], loss_d=d_loss[-1])
    
    return img, report, chart_data

# --- ULTIMATE INTERFACE DESIGN ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate")) as demo:
    gr.Markdown("# 🌌 AETHER-SYNTH PRO | Autonomous Neural Lab v2.0")
    gr.Markdown("🧪 **Status:** Enterprise Grade Research Environment Active")

    with gr.Tabs():
        with gr.TabItem("🚀 Research Terminal"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### ⚙️ Hyperparameters")
                    seed = gr.Number(label="Neural Seed", value=42)
                    lr = gr.Slider(0.0001, 0.01, label="Simulated Learning Rate", value=0.0002)
                    batch = gr.Slider(1, 64, label="Sample Batch Size", step=1, value=1)
                    run_btn = gr.Button("🔥 Execute Synthesis", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Row():
                        out_img = gr.Image(label="Current Artifact", height=300)
                        loss_plot = gr.LinePlot(label="Live Training Convergence", x="Epoch", y="Gen Loss", tooltip=["Epoch", "Gen Loss", "Disc Loss"], height=300)
            
            gr.Markdown("### 🤖 Cognitive Audit & XAI Reasoning")
            audit_report = gr.Markdown("Waiting for neural trigger...")

        with gr.TabItem("🏗️ System Internals"):
            with gr.Row():
                gr.Code(str(gen), label="Generator (G) Architecture", language="python")
                gr.Code(str(disc), label="Discriminator (D) Architecture", language="python")

    run_btn.click(fn=live_research_pipeline, inputs=[seed, lr, batch], outputs=[out_img, audit_report, loss_plot])

if __name__ == "__main__":
    demo.launch(share=True)
