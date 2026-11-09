"""PixelLab: broad Pillow + Matplotlib image studio. Run: streamlit run main.py"""
import io
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
import numpy as np
import streamlit as st
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat

st.set_page_config(page_title="PixelLab Studio",
                   page_icon="space_pointer.png",
                   layout="wide")

st.markdown("""<style>.block-container{padding-top:1.2rem;max-width:1700px}div[data-testid="stMetric"]{background:#18231f10;padding:12px;border-radius:12px}</style>""", unsafe_allow_html=True)
st.title("🎨 PixelLab Studio")
st.caption("Pillow pixel editing • Matplotlib color science and visualizations • multi-stage, non-destructive preview")

MAX_PIXELS = 30_000_000  # Guard large files for responsive hosted apps.

@st.cache_data(show_spinner=False)
def load_image(data: bytes):
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"Image exceeds {MAX_PIXELS:,}"
                             f" pixels. Resize it before uploading.")
        return ImageOps.exif_transpose(im).convert("RGBA")

def clamp8(a):
    return np.uint8(np.clip(a, 0, 255))

def rgb_array(im):
    return np.asarray(im.convert("RGB"), dtype=np.uint8)

def to_png(im):
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()

def convert_export(im, fmt, quality=92):
    out = io.BytesIO()
    if fmt in ("JPEG", "PDF"):
        canvas = Image.new("RGB", im.size, "white")
        canvas.paste(im.convert("RGBA"), mask=im.getchannel("A"))
        im = canvas
    args = {"quality": quality, "optimize": True} if fmt in ("JPEG", "WEBP") else {}
    im.save(out, format=fmt, **args)
    return out.getvalue()

def mpl_norm(kind, lo, hi, gamma):
    if kind == "Log":
        return colors.LogNorm(vmin=max(lo, 0.01), vmax=max(hi, max(lo, 0.01) + .01))
    if kind == "Power":
        return colors.PowerNorm(gamma=gamma, vmin=lo, vmax=hi)
    if kind == "SymLog":
        return colors.SymLogNorm(linthresh=10, vmin=lo, vmax=hi)
    return colors.Normalize(vmin=lo, vmax=hi)

def render_plot(fig, fmt="png", dpi=140):
    data = io.BytesIO()
    fig.savefig(data, format=fmt, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return data.getvalue()

def apply_pillow(im, p):
    im = im.copy()
    # Geometry
    if p["resize"]:
        w, h = im.size
        if p["aspect"]:
            target = (max(1, int(w * p["scale"])), max(1, int(h * p["scale"])))
        else:
            target = (p["width"], p["height"])
        if target[0] * target[1] > MAX_PIXELS:
            raise ValueError("Resize would exceed maximum pixel limit.")
        im = im.resize(target, getattr(Image.Resampling, p["resample"]))
    if p["crop"]:
        w, h = im.size
        left, top, right, bottom = p["crop_values"]
        rect = (int(w*left/100), int(h*top/100), int(w*right/100), int(h*bottom/100))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError("Crop right/bottom must exceed left/top.")
        im = im.crop(rect)
    if p["rotate"]:
        im = im.rotate(p["angle"], expand=p["expand"], resample=Image.Resampling.BICUBIC)
    if p["flip_h"]: im = ImageOps.mirror(im)
    if p["flip_v"]: im = ImageOps.flip(im)
    # Enhancements operate on RGB while preserving alpha.
    alpha = im.getchannel("A")
    rgb = im.convert("RGB")
    for key, cls in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast),
                     ("saturation", ImageEnhance.Color), ("sharpness", ImageEnhance.Sharpness)):
        if p[key] != 1.0:
            rgb = cls(rgb).enhance(p[key])
    if p["autocontrast"]: rgb = ImageOps.autocontrast(rgb, cutoff=p["cutoff"])
    if p["equalize"]: rgb = ImageOps.equalize(rgb)
    if p["invert"]: rgb = ImageOps.invert(rgb)
    if p["solarize"]: rgb = ImageOps.solarize(rgb, threshold=p["solar_threshold"])
    if p["posterize"]: rgb = ImageOps.posterize(rgb, bits=p["bits"])
    if p["grayscale"]: rgb = ImageOps.grayscale(rgb).convert("RGB")
    if p["colorize"]:
        rgb = ImageOps.colorize(ImageOps.grayscale(rgb), black=p["shadow_color"], white=p["highlight_color"])
    if p["filter"] != "None":
        filter_name = p["filter"]
        if filter_name == "GaussianBlur": fil = ImageFilter.GaussianBlur(radius=p["radius"])
        elif filter_name == "BoxBlur": fil = ImageFilter.BoxBlur(radius=p["radius"])
        elif filter_name == "UnsharpMask": fil = ImageFilter.UnsharpMask(radius=p["radius"], percent=p["unsharp_percent"], threshold=p["threshold"])
        elif filter_name in ("MedianFilter", "MinFilter", "MaxFilter", "ModeFilter"):
            fil = getattr(ImageFilter, filter_name)(size=p["kernel"])
        else: fil = getattr(ImageFilter, filter_name)
        rgb = rgb.filter(fil)
    if p["custom_kernel"]:
        k = np.array(p["matrix"], dtype=float).reshape((3, 3))
        scale = float(np.sum(k)) or 1.0
        rgb = rgb.filter(ImageFilter.Kernel((3,3), k.flatten().tolist(), scale=scale))
    if p["channel"] != "RGB":
        r, g, b = rgb.split()
        zero = Image.new("L", rgb.size, 0)
        if p["channel"] == "Red": rgb = Image.merge("RGB", (r,zero,zero))
        elif p["channel"] == "Green": rgb = Image.merge("RGB", (zero,g,zero))
        elif p["channel"] == "Blue": rgb = Image.merge("RGB", (zero,zero,b))
        elif p["channel"] == "Luminance": rgb = ImageOps.grayscale(rgb).convert("RGB")
    im = rgb.convert("RGBA")
    im.putalpha(alpha)
    if p["border"]:
        im = ImageOps.expand(im, border=p["border_size"], fill=p["border_color"])
    return im

def apply_overlay(im, p, overlay_bytes):
    im = im.copy()
    if overlay_bytes:
        over = load_image(overlay_bytes).copy()
        if p["overlay_fit"]: over = over.resize(im.size, Image.Resampling.LANCZOS)
        else:
            targetw = max(1,int(im.width*p["overlay_scale"]/100))
            over.thumbnail((targetw, max(1,im.height*3)), Image.Resampling.LANCZOS)
        a = over.getchannel("A").point(lambda x: round(x * p["overlay_opacity"] / 100))
        over.putalpha(a)
        x = int((im.width-over.width)*p["overlay_x"]/100)
        y = int((im.height-over.height)*p["overlay_y"]/100)
        layer = Image.new("RGBA", im.size)
        layer.paste(over, (x,y))
        if p["blend"] == "Normal": im = Image.alpha_composite(im, layer)
        else:
            base_rgb, layer_rgb = im.convert("RGB"), layer.convert("RGB")
            method = {"Multiply": ImageChops.multiply, "Screen": ImageChops.screen,
                      "Difference": ImageChops.difference, "Add": ImageChops.add,
                      "Subtract": ImageChops.subtract, "Lighter": ImageChops.lighter,
                      "Darker": ImageChops.darker}[p["blend"]]
            composed = method(base_rgb, layer_rgb).convert("RGBA")
            im = Image.composite(composed, im, layer.getchannel("A"))
    if p["text"] or p["draw_shape"] != "None":
        draw = ImageDraw.Draw(im)
        x = round(im.width*p["draw_x"]/100)
        y = round(im.height*p["draw_y"]/100)
        w = round(im.width*p["draw_w"]/100)
        h = round(im.height*p["draw_h"]/100)
        color = p["draw_color"]
        shape = p["draw_shape"]
        if shape == "Rectangle": draw.rectangle((x,y,x+w,y+h), outline=color, width=p["stroke"])
        if shape == "Filled rectangle": draw.rectangle((x,y,x+w,y+h), fill=color)
        if shape == "Ellipse": draw.ellipse((x,y,x+w,y+h), outline=color, width=p["stroke"])
        if shape == "Filled ellipse": draw.ellipse((x,y,x+w,y+h), fill=color)
        if shape == "Line": draw.line((x,y,x+w,y+h), fill=color, width=p["stroke"])
        if p["text"]:
            font_path = p["font_path"]
            try: font = ImageFont.truetype(font_path, p["font_size"]) if font_path else ImageFont.load_default(size=p["font_size"])
            except (OSError, TypeError, ValueError): font = ImageFont.load_default()
            draw.multiline_text((x,y), p["text"], font=font, fill=color, spacing=5, stroke_width=p["text_stroke"], stroke_fill=p["stroke_color"])
    return im

def apply_mpl_colormap(im, p):
    if not p["apply_cmap"]: return im
    arr = rgb_array(im).astype(np.float32)
    if p["channel_map"] == "R": gray = arr[:,:,0]
    elif p["channel_map"] == "G": gray = arr[:,:,1]
    elif p["channel_map"] == "B": gray = arr[:,:,2]
    else: gray = arr @ np.array([.2126,.7152,.0722], dtype=np.float32)
    norm = mpl_norm(p["norm"], p["vmin"], p["vmax"], p["gamma"])
    rgba = clamp8(colormaps[p["cmap"]](norm(gray))*255)
    recolored = Image.fromarray(rgba, "RGBA")
    recolored.putalpha(im.getchannel("A"))
    return Image.blend(im, recolored, p["cmap_strength"] / 100)

with st.sidebar:
    st.image("softwareLogo.png", width=300)
    st.space(20)
    upload = st.file_uploader("Base image", type=["jpg","jpeg","png","webp","bmp","tif","tiff","gif"])
    overlay_upload = st.file_uploader("Optional overlay / second image", type=["png","jpg","jpeg","webp"])
    st.caption("Operations run in order: Geometry → Color → Filters → Matplotlib colormap → Overlay/Draw")
    st.divider()
    st.header("⚙️ Preview")
    compare = st.checkbox("Side-by-side comparison", value=True)
    show_rgba = st.checkbox("Show transparency checkerboard", value=False)

if upload is None:
    st.info("Upload a base image in the sidebar to begin. Optional: upload a second image for compositing.")
    st.stop()
try:
    base = load_image(upload.getvalue())
except Exception as exc:
    st.error(f"Could not read image: {exc}")
    st.stop()

p = {}
tabs = st.tabs(["📐 Geometry", "🌈 Color & Tone", "✨ Filters", "🎨 Matplotlib Maps", "🖌️ Layers & Drawing", "📊 Analysis", "📤 Export"])
with tabs[0]:
    a,b,c = st.columns(3)
    with a:
        st.subheader("Resizing")
        p["resize"] = st.checkbox("Enable resize")
        p["aspect"] = st.checkbox("Maintain aspect ratio", True)
        p["scale"] = st.slider("Scale", 0.05, 4.0, 1.0, .05)
        p["width"] = st.number_input("Width (px)", 1, 10000, base.width)
        p["height"] = st.number_input("Height (px)", 1, 10000, base.height)
        p["resample"] = st.selectbox("Resampling", ["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST", "BOX", "HAMMING"])
    with b:
        st.subheader("Crop & orientation")
        p["crop"] = st.checkbox("Enable percentage crop")
        cl = st.slider("Left %",0,99,0); ct = st.slider("Top %",0,99,0)
        cr = st.slider("Right %",1,100,100); cb = st.slider("Bottom %",1,100,100)
        p["crop_values"]=(cl,ct,cr,cb)
        p["rotate"] = st.checkbox("Enable rotation")
        p["angle"] = st.slider("Angle (degrees)",-180,180,0)
        p["expand"] = st.checkbox("Expand canvas for rotation",True)
    with c:
        st.subheader("Flips & framing")
        p["flip_h"] = st.checkbox("Flip horizontally")
        p["flip_v"] = st.checkbox("Flip vertically")
        p["border"] = st.checkbox("Add border")
        p["border_size"] = st.slider("Border width",1,200,15)
        p["border_color"] = st.color_picker("Border color","#ffffff")

with tabs[1]:
    a,b,c = st.columns(3)
    with a:
        st.subheader("ImageEnhance")
        for name in ("brightness","contrast","saturation","sharpness"):
            p[name] = st.slider(name.title(),0.0,4.0,1.0,.05)
    with b:
        st.subheader("ImageOps")
        p["autocontrast"] = st.checkbox("Auto contrast")
        p["cutoff"] = st.slider("Autocontrast cutoff %",0,25,1)
        p["equalize"] = st.checkbox("Histogram equalization")
        p["invert"] = st.checkbox("Invert colors")
        p["solarize"] = st.checkbox("Solarize")
        p["solar_threshold"] = st.slider("Solarization threshold",0,255,128)
        p["posterize"] = st.checkbox("Posterize")
        p["bits"] = st.slider("Bits per channel",1,8,4)
    with c:
        st.subheader("Channels & tint")
        p["grayscale"] = st.checkbox("Grayscale")
        p["channel"] = st.selectbox("Isolate channel",["RGB","Red","Green","Blue","Luminance"])
        p["colorize"] = st.checkbox("Duotone colorize")
        p["shadow_color"] = st.color_picker("Shadow color","#191957")
        p["highlight_color"] = st.color_picker("Highlight color","#f8cf78")

with tabs[2]:
    a,b = st.columns(2)
    choices = ["None","BLUR","CONTOUR","DETAIL","EDGE_ENHANCE","EDGE_ENHANCE_MORE","EMBOSS","FIND_EDGES","SHARPEN","SMOOTH","SMOOTH_MORE","GaussianBlur","BoxBlur","UnsharpMask","MedianFilter","MinFilter","MaxFilter","ModeFilter"]
    with a:
        st.subheader("Built-in Pillow filters")
        p["filter"] = st.selectbox("Filter",choices)
        p["radius"] = st.slider("Blur / unsharp radius",0.0,25.0,2.0,.5)
        p["unsharp_percent"] = st.slider("Unsharp amount",0,500,150)
        p["threshold"] = st.slider("Unsharp threshold",0,50,3)
        p["kernel"] = st.select_slider("Rank filter kernel", options=[3,5,7,9],value=3)
    with b:
        st.subheader("Custom convolution kernel")
        p["custom_kernel"] = st.checkbox("Apply 3×3 kernel after built-in filter")
        preset = st.selectbox("Kernel preset",["Identity","Edge","Sharpen","Emboss","Custom"])
        presets = {"Identity":"0,0,0,0,1,0,0,0,0", "Edge":"-1,-1,-1,-1,8,-1,-1,-1,-1", "Sharpen":"0,-1,0,-1,5,-1,0,-1,0", "Emboss":"-2,-1,0,-1,1,1,0,1,2"}
        raw = st.text_input("9 comma-separated coefficients",value=presets.get(preset,"0,-1,0,-1,5,-1,0,-1,0"))
        try:
            p["matrix"] = [float(x.strip()) for x in raw.split(",")]
            assert len(p["matrix"]) == 9
        except (ValueError, AssertionError):
            p["matrix"]=[0,0,0,0,1,0,0,0,0]
            if p["custom_kernel"]: st.warning("Provide exactly nine numeric coefficients.")

with tabs[3]:
    a,b,c = st.columns(3)
    with a:
        st.subheader("Scalar colormapping")
        p["apply_cmap"] = st.checkbox("Apply Matplotlib colormap to image")
        cmap_names = sorted(colormaps)
        p["cmap"] = st.selectbox("Colormap",cmap_names,index=cmap_names.index("viridis"))
        p["channel_map"] = st.selectbox("Source values",["Luminance","R","G","B"])
        p["cmap_strength"] = st.slider("Colormap opacity %",0,100,100)
    with b:
        st.subheader("Normalize")
        p["norm"] = st.selectbox("Normalization",["Linear","Log","Power","SymLog"])
        p["vmin"] = st.slider("Minimum intensity",0,254,0)
        p["vmax"] = st.slider("Maximum intensity",1,255,255)
        p["gamma"] = st.slider("Power gamma",0.1,5.0,1.0,.1)
        if p["vmax"] <= p["vmin"]: st.warning("Max intensity must exceed min.")
    with c:
        st.subheader("Display & rendering")
        interp_options=["none","nearest","bilinear","bicubic","spline16","spline36","hanning","hamming","hermite","kaiser","quadric","catrom","gaussian","bessel","mitchell","sinc","lanczos","blackman"]
        interpolation=st.selectbox("imshow interpolation",interp_options,index=2)
        origin=st.selectbox("Image origin",["upper","lower"])
        aspect=st.selectbox("Image aspect",["equal","auto"])
        show_colorbar=st.checkbox("Show scalar colorbar",True)
        plot_dpi=st.slider("Matplotlib export DPI",72,400,160,8)

with tabs[4]:
    a,b,c=st.columns(3)
    with a:
        st.subheader("Second-image compositing")
        overlay_bytes=overlay_upload.getvalue() if overlay_upload else None
        p["overlay_fit"] = st.checkbox("Fit overlay to full canvas")
        p["overlay_scale"] = st.slider("Overlay width %",5,100,35)
        p["overlay_opacity"] = st.slider("Overlay opacity %",0,100,85)
        p["blend"] = st.selectbox("Blend mode",["Normal","Multiply","Screen","Difference","Add","Subtract","Lighter","Darker"])
        p["overlay_x"] = st.slider("Overlay horizontal placement %",0,100,50)
        p["overlay_y"] = st.slider("Overlay vertical placement %",0,100,50)
        if not overlay_upload: st.caption("Upload an overlay in the sidebar to enable compositing.")
    with b:
        st.subheader("Vector drawing")
        p["draw_shape"] = st.selectbox("Shape",["None","Rectangle","Filled rectangle","Ellipse","Filled ellipse","Line"])
        p["draw_x"] = st.slider("Left %",0,100,10,key="drawx")
        p["draw_y"] = st.slider("Top %",0,100,10,key="drawy")
        p["draw_w"] = st.slider("Shape width %",0,100,45)
        p["draw_h"] = st.slider("Shape height %",0,100,30)
        p["draw_color"] = st.color_picker("Draw/text color","#ffcc33")
        p["stroke"] = st.slider("Line width px",1,40,5)
    with c:
        st.subheader("Text & annotation")
        p["text"] = st.text_area("Overlay text (optional)",height=110)
        p["font_size"] = st.slider("Font size px",10,160,36)
        p["font_path"] = st.text_input("Optional server font path (.ttf)",placeholder="/path/to/font.ttf")
        p["text_stroke"] = st.slider("Text outline width",0,12,0)
        p["stroke_color"] = st.color_picker("Text outline color","#000000")

try:
    result = apply_pillow(base,p)
    if p["vmax"] <= p["vmin"]: raise ValueError("Matplotlib maximum intensity must exceed minimum.")
    result = apply_mpl_colormap(result,p)
    result = apply_overlay(result,p,overlay_bytes)
except Exception as exc:
    st.error(f"Processing error: {exc}")
    st.stop()

st.divider()
st.subheader("🖼️ Live result")
if compare:
    original_col, result_col = st.columns(2)
    with original_col:
        st.caption("Original")
        st.image(base, use_container_width=True)
    with result_col:
        st.caption("Edited")
        st.image(result, use_container_width=True)
else:
    st.image(result, use_container_width=True)

x,y=result.size
m1,m2,m3,m4=st.columns(4)
m1.metric("Width",f"{x:,} px"); m2.metric("Height",f"{y:,} px")
m3.metric("Pixels",f"{x*y:,}"); m4.metric("Mode",result.mode)
if show_rgba:
    tile=20
    yy,xx=np.indices((result.height,result.width))
    checks=(((xx//tile)+(yy//tile))%2*38+202).astype(np.uint8)
    bg=Image.fromarray(np.repeat(checks[:,:,None],3,axis=2),"RGB").convert("RGBA")
    st.image(Image.alpha_composite(bg,result),caption="Alpha composited over checkerboard",use_container_width=True)

with tabs[5]:
    st.subheader("Matplotlib image diagnostics")
    rgb=rgb_array(result)
    gray=np.asarray(ImageOps.grayscale(result.convert("RGB")),dtype=np.float32)
    visualization=st.selectbox("Chart type",["RGB histograms","Grayscale histogram","Channel images","imshow (interpolation / origin)","False-color heatmap","Pixel value profile","Image statistics","Fourier magnitude spectrum"])
    fig, ax = plt.subplots(figsize=(8,4.5),layout="constrained")
    if visualization == "RGB histograms":
        for channel,label in enumerate(("Red","Green","Blue")):
            ax.hist(rgb[:,:,channel].flatten(),bins=64,alpha=.45,label=label,color=label.lower())
        ax.legend(); ax.set(xlabel="Pixel intensity (0–255)",ylabel="Count",title="RGB distribution")
    elif visualization == "Grayscale histogram":
        ax.hist(gray.ravel(),bins=64,color="dimgray"); ax.set(xlabel="Intensity",ylabel="Count",title="Luminance histogram")
    elif visualization == "Channel images":
        plt.close(fig)
        fig, axes = plt.subplots(1,3,figsize=(11,3.3),layout="constrained")
        for i,label in enumerate(("Red","Green","Blue")):
            axes[i].imshow(rgb[:,:,i],cmap="gray",vmin=0,vmax=255,interpolation=interpolation,origin=origin)
            axes[i].set_title(label); axes[i].axis("off")
    elif visualization == "imshow (interpolation / origin)":
        ax.imshow(rgb,interpolation=interpolation,origin=origin,aspect=aspect)
        ax.set_title("Matplotlib RGB image renderer"); ax.axis("off")
    elif visualization == "False-color heatmap":
        art=ax.imshow(gray,cmap=p["cmap"],norm=mpl_norm(p["norm"],p["vmin"],p["vmax"],p["gamma"]),interpolation=interpolation,origin=origin,aspect=aspect)
        if show_colorbar: fig.colorbar(art,ax=ax,label="Intensity")
        ax.set_title("Matplotlib intensity heatmap"); ax.axis("off")
    elif visualization == "Pixel value profile":
        line_y=st.slider("Profile row %",0,100,50)
        yy=min(gray.shape[0]-1,round((gray.shape[0]-1)*line_y/100))
        for i,name in enumerate(("Red","Green","Blue")):
            ax.plot(rgb[yy,:,i],label=name,color=name.lower(),alpha=.75)
        ax.legend(); ax.set(xlabel="X pixel",ylabel="Intensity",title=f"Row {yy} profile")
    elif visualization == "Image statistics":
        stats=ImageStat.Stat(result.convert("RGB"))
        st.json({"mean":dict(zip("RGB",map(float,stats.mean))),"stddev":dict(zip("RGB",map(float,stats.stddev))),"extrema":dict(zip("RGB",stats.extrema)),"bbox":result.getbbox()})
        ax.bar(["R","G","B"],stats.mean,yerr=stats.stddev,color=["red","green","blue"],alpha=.6)
        ax.set(ylabel="Mean ± standard deviation",ylim=(0,255),title="Channel statistics")
    else:
        # Limit FFT input to remain memory/responsiveness-friendly.
        small=Image.fromarray(clamp8(gray)).copy()
        small.thumbnail((768,768))
        signal=np.asarray(small,dtype=np.float32)
        spectrum=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(signal))))
        art=ax.imshow(spectrum,cmap=p["cmap"],interpolation="nearest")
        if show_colorbar: fig.colorbar(art,ax=ax)
        ax.set_title("Log Fourier magnitude spectrum"); ax.axis("off")
    chart_png=render_plot(fig,"png",plot_dpi)
    st.image(chart_png,caption=visualization,use_container_width=True)
    st.download_button("Download chart PNG",chart_png,file_name="analysis.png",mime="image/png")

with tabs[6]:
    st.subheader("Pillow image exports")
    fmt=st.selectbox("Save as",["PNG","JPEG","WEBP","TIFF","BMP","PDF"])
    quality=st.slider("JPEG / WebP quality",10,100,92)
    try:
        edited_bytes=convert_export(result,fmt,quality)
        extension={"JPEG":"jpg","TIFF":"tiff"}.get(fmt,fmt.lower())
        st.download_button(f"Download edited {fmt}",edited_bytes,file_name=f"edited.{extension}",mime={"PNG":"image/png","JPEG":"image/jpeg","WEBP":"image/webp","TIFF":"image/tiff","BMP":"image/bmp","PDF":"application/pdf"}[fmt],type="primary")
    except Exception as exc:
        st.error(f"Export failed: {exc}")
    st.subheader("Matplotlib rendered image")
    mpl_export_cmap=st.checkbox("Render grayscale as Matplotlib scalar colormap",False)
    mpl_export_colorbar=st.checkbox("Include colorbar in rendered figure",False)
    export_fmt=st.selectbox("Figure format",["png","svg","pdf"],key="figure_fmt")
    if st.button("Generate Matplotlib figure"):
        fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
        if mpl_export_cmap:
            arr=np.asarray(ImageOps.grayscale(result.convert("RGB")),dtype=float)
            m=ax.imshow(arr,cmap=p["cmap"],norm=mpl_norm(p["norm"],p["vmin"],p["vmax"],p["gamma"]),interpolation=interpolation,origin=origin,aspect=aspect)
            if mpl_export_colorbar: fig.colorbar(m,ax=ax)
        else: ax.imshow(result,interpolation=interpolation,origin=origin,aspect=aspect)
        ax.axis("off")
        st.session_state["mpl_export"] = render_plot(fig,export_fmt,plot_dpi)
        st.session_state["mpl_export_fmt"] = export_fmt
    if st.session_state.get("mpl_export"):
        st.download_button("Download rendered figure",st.session_state["mpl_export"],file_name=f"matplotlib_render.{st.session_state['mpl_export_fmt']}")
    st.caption("Matplotlib figure exports re-render on demand; Pillow exports preserve the edited image's pixel dimensions. SVG/PDF plots may raster-embed images.")

with st.expander("📚 Methods included & limitations"):
    st.markdown("""
    **Pillow:** Image.open/save, resize and six resampling modes, crop, rotate, flips, alpha compositing,
    blend modes via ImageChops, ImageEnhance (brightness/contrast/color/sharpness), ImageOps (autocontrast,
    equalize, invert, solarize, posterize, grayscale, colorize, expand), common ImageFilter kernels,
    BoxBlur/GaussianBlur/UnsharpMask, rank filters, custom convolution, ImageDraw shapes/text, ImageStat,
    multiple image export formats.

    **Matplotlib:** registered colormaps, image scalar mapping, Normalize / LogNorm / PowerNorm / SymLogNorm,
    imshow interpolation modes, aspect/origin, colorbars, channel/histogram/profile analyses, FFT view,
    and savefig PNG/SVG/PDF with DPI controls.

    This is an extensive **practical editor**, not literally every public API in both libraries (which include
    scientific plotting, 3D graphics, font internals, codecs, ImageCms profiles, animations, and much more).
    Matplotlib is primarily a visualization/rendering library, not a replacement for pixel editing.
    Very large images consume memory, and some export codecs depend on the installed Pillow build.
    """)
