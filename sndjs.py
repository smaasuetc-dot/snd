# Beam Analysis Pro: Full Robust Streamlit App
# Complete Version with Plotly Graphs, Reactions, and State Management

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
from reportlab.lib.utils import ImageReader  # make sure this import is at top


st.set_page_config(page_title="Beam Analysis Pro", layout="wide")
st.title("🔧 SnD Beam Analysis")

# --- Session State Init ---
if "units" not in st.session_state:
    st.session_state.units = "SI"
for key in ["supports", "point_loads", "udls", "vdls", "moments", "reactions"]:
    if key not in st.session_state:
        st.session_state[key] = []

# --- Units Toggle ---
unit_opts = {"SI": {"F": "kN", "L": "m"}, "Imperial": {"F": "lb", "L": "ft"}}
st.sidebar.selectbox("Units", ["SI", "Imperial"], key="units")
U = unit_opts[st.session_state.units]

# --- Beam Setup ---
st.sidebar.header("📐 Beam Setup")
beam_length = st.sidebar.number_input(f"Beam Length ({U['L']})", min_value=1.0, value=11.0, step=0.1)

# --- Supports ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧱 Supports")
num_supports = st.sidebar.slider("Number of Supports", 1, 5, 2)
for i in range(num_supports):
    col1, col2 = st.sidebar.columns(2)
    pos = col1.number_input(f"Support {i+1} Pos ({U['L']})", key=f"supp_pos_{i}", min_value=0.0, max_value=beam_length, value=float(i*(beam_length/(num_supports-1))))
    typ = col2.selectbox(f"Type", ["Pinned", "Roller", "Fixed"], key=f"supp_type_{i}")
    if len(st.session_state.supports) < num_supports:
        st.session_state.supports.append({"pos": pos, "type": typ})
    else:
        st.session_state.supports[i] = {"pos": pos, "type": typ}

# --- Load Inputs ---
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Loads")
with st.sidebar.expander("➕ Point Load"):
    mag = st.number_input(f"Magnitude ({U['F']})", value=10.0)
    pos = st.number_input(f"Position ({U['L']})", value=5.0)
    if st.button("Add Point Load"):
        st.session_state.point_loads.append({"mag": mag, "pos": pos})

with st.sidebar.expander("➕ UDL"):
    inten = st.number_input(f"Intensity ({U['F']}/{U['L']})", value=3.0)
    start = st.number_input(f"Start Pos ({U['L']})", value=2.0)
    end = st.number_input(f"End Pos ({U['L']})", value=6.0)
    if st.button("Add UDL"):
        st.session_state.udls.append({"start": start, "end": end, "intensity": inten})

with st.sidebar.expander("➕ VDL"):
    vstart = st.number_input(f"Start Pos ({U['L']})", value=1.0)
    vend = st.number_input(f"End Pos ({U['L']})", value=4.0)
    w1 = st.number_input(f"w1 ({U['F']}/{U['L']})", value=0.0)
    w2 = st.number_input(f"w2 ({U['F']}/{U['L']})", value=5.0)
    if st.button("Add VDL"):
        st.session_state.vdls.append({"start": vstart, "end": vend, "w1": w1, "w2": w2})

with st.sidebar.expander("➕ Moment Load"):
    mpos = st.number_input(f"Position ({U['L']})", value=3.0)
    mval = st.number_input(f"Moment ({U['F']}×{U['L']})", value=10.0)
    if st.button("Add Moment"):
        st.session_state.moments.append({"pos": mpos, "val": mval})

# --- Display Loads ---
st.subheader("📋 Current Loads")
col1, col2 = st.columns(2)
with col1:
    st.write("### Point Loads")
    st.table(st.session_state.point_loads)
    st.write("### UDLs")
    st.table(st.session_state.udls)
with col2:
    st.write("### VDLs")
    st.table(st.session_state.vdls)
    st.write("### Moments")
    st.table(st.session_state.moments)

# --- Reaction Solver ---
supports_sorted = sorted(st.session_state.supports, key=lambda s: s["pos"])
support_positions = [s["pos"] for s in supports_sorted]
support_types = [s["type"] for s in supports_sorted]

A_pos = support_positions[0]
B_pos = support_positions[-1]

# Total downward force
F_total = sum([pl["mag"] for pl in st.session_state.point_loads])
for udl in st.session_state.udls:
    F_total += udl["intensity"] * (udl["end"] - udl["start"])
for vdl in st.session_state.vdls:
    w_avg = (vdl["w1"] + vdl["w2"]) / 2
    F_total += w_avg * (vdl["end"] - vdl["start"])

# Moment about A
M_A = 0.0
for pl in st.session_state.point_loads:
    M_A += pl["mag"] * (pl["pos"] - A_pos)
for udl in st.session_state.udls:
    L = udl["end"] - udl["start"]
    M_A += udl["intensity"] * L * ((udl["start"] + udl["end"])/2 - A_pos)
for vdl in st.session_state.vdls:
    L = vdl["end"] - vdl["start"]
    w_avg = (vdl["w1"] + vdl["w2"]) / 2
    M_A += w_avg * L * ((vdl["start"] + vdl["end"])/2 - A_pos)

# Solve reactions (assume 2 supports)
if len(support_positions) == 2:
    RB = M_A / (B_pos - A_pos)
    RA = F_total - RB
    st.session_state.reactions = [(A_pos, RA), (B_pos, RB)]

# --- Display Reactions ---
st.subheader("🧱 Support Reactions")
for pos, R in st.session_state.reactions:
    st.write(f"Reaction at {pos:.2f} {U['L']}: {R:.2f} {U['F']}")

# --- Diagram Plotting ---
st.subheader("📊 SFD and BMD")

x = np.linspace(0, beam_length, 500)
shear = np.zeros_like(x)
moment = np.zeros_like(x)

for pos, R in st.session_state.reactions:
    shear[x >= pos] += R
    moment[x >= pos] += R * (x[x >= pos] - pos)

for pl in st.session_state.point_loads:
    shear[x >= pl['pos']] -= pl['mag']
    moment[x >= pl['pos']] -= pl['mag'] * (x[x >= pl['pos']] - pl['pos'])

for udl in st.session_state.udls:
    idx = (x >= udl['start']) & (x <= udl['end'])
    L = udl['end'] - udl['start']
    shear[idx] -= udl['intensity'] * (x[idx] - udl['start'])
    moment[idx] -= 0.5 * udl['intensity'] * (x[idx] - udl['start']) ** 2
    shear[x > udl['end']] -= udl['intensity'] * L
    moment[x > udl['end']] -= udl['intensity'] * L * (x[x > udl['end']] - (udl['start'] + L / 2))

for vdl in st.session_state.vdls:
    idx = (x >= vdl['start']) & (x <= vdl['end'])
    a = (vdl['w2'] - vdl['w1']) / (vdl['end'] - vdl['start'])
    b = vdl['w1']
    xi = x[idx] - vdl['start']
    w = a * xi + b
    shear[idx] -= np.cumsum(w) * (x[1] - x[0])
    moment[idx] -= np.cumsum(shear[idx]) * (x[1] - x[0])

for mo in st.session_state.moments:
    moment[x >= mo['pos']] += mo['val']

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=shear, mode='lines', name='Shear Force', line=dict(color='crimson', width=3)))
fig.add_trace(go.Scatter(x=x, y=moment, mode='lines', name='Bending Moment', line=dict(color='royalblue', width=3)))
fig.update_layout(title="Shear Force & Bending Moment Diagrams", xaxis_title=f"Beam Length ({U['L']})", yaxis_title=f"Force / Moment ({U['F']}, {U['F']}×{U['L']})", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Summary ---
st.success("✅ Beam analysis updated with support reaction solver and improved diagrams. Ready for deflection and export modules.")

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
from PIL import Image
import plotly.io as pio

# --- Export Button ---
if st.button("📤 Export to PDF"):
    # Save plotly chart to image
    img_buf = BytesIO()
    fig.write_image(img_buf, format="png", engine="kaleido", width=800, height=400)
    img_buf.seek(0)
    img = Image.open(img_buf)

    # Create PDF buffer
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Beam Analysis Report")

    # Beam Info
    c.setFont("Helvetica", 12)
    y = height - 90
    c.drawString(50, y, f"Beam Length: {beam_length} {U['L']}")

    # Supports
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Supports:")
    c.setFont("Helvetica", 11)
    for s in st.session_state.supports:
        y -= 18
        c.drawString(60, y, f"{s['type']} at {s['pos']} {U['L']}")

    # Loads
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Point Loads:")
    c.setFont("Helvetica", 11)
    for pl in st.session_state.point_loads:
        y -= 18
        c.drawString(60, y, f"{pl['mag']} {U['F']} at {pl['pos']} {U['L']}")

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "UDLs:")
    c.setFont("Helvetica", 11)
    for udl in st.session_state.udls:
        y -= 18
        c.drawString(60, y, f"{udl['intensity']} {U['F']}/{U['L']} from {udl['start']} to {udl['end']}")

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "VDLs:")
    c.setFont("Helvetica", 11)
    for vdl in st.session_state.vdls:
        y -= 18
        c.drawString(60, y, f"{vdl['w1']}→{vdl['w2']} {U['F']}/{U['L']} from {vdl['start']} to {vdl['end']}")

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Moments:")
    c.setFont("Helvetica", 11)
    for mo in st.session_state.moments:
        y -= 18
        c.drawString(60, y, f"{mo['val']} {U['F']}×{U['L']} at {mo['pos']} {U['L']}")

    # Add chart image
    y -= 250
    c.drawImage(ImageReader(img), 50, y, width=500, preserveAspectRatio=True)

    # Save and display download button
    c.showPage()
    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="Beam_Analysis_Report.pdf",
        mime="application/pdf"
    )
