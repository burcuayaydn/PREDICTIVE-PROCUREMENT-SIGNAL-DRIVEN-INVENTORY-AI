# ==============================================================================
# PROJE: OPTICHAIN | AI-DRIVEN PROCUREMENT CONTROL TOWER
# MODÜL: STREAMLIT_APP.PY
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Konsol uyarılarını gizleme
warnings.filterwarnings('ignore')

# 1. SAYFA MİMARİSİ VE SEKMELİ UI OPTİMİZASYONU
st.set_page_config(
    page_title="OptiChain | Signal-Driven Procurement Control Tower",
    page_icon="📊",
    layout="wide"
)

# Tasarım, Kontrast ve Belirgin Metrik Alt Yazı CSS Katmanı
st.markdown("""
    <style>
        header, [data-testid="stHeader"], [data-testid="stDecoration"] {
            display: none !important;
        }
        .stApp {
            background-color: #0e1117 !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #06080c !important;
            border-right: 1px solid #21262d;
            min-width: 260px !important;
            max-width: 260px !important;
        }
        .executive-box {
            background-color: #1c212c;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #00ffd0;
            margin-bottom: 15px;
            border: 1px solid #30363d;
        }
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
            margin-top: 5px !important;
            margin-bottom: 5px !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #ffffff !important; 
            font-size: 15px !important; 
            font-weight: 700 !important;
        }

        /* SİLİK METRİK ALT YAZILARINI (DELTA) BEYAZ VE NET YAPMA SİHREBBAZI */
        [data-testid="stMetricDelta"] {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricDelta"] svg {
            fill: #00ffd0 !important;
        }

        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #06080c;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid #21262d;
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px;
            background-color: #1c212c;
            border-radius: 6px;
            color: #c9d1d9 !important;
            font-weight: 600;
            padding: 0px 16px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #00ffd0 !important;
            color: #06080c !important;
        }
        [data-testid="stMetricValue"] {
            color: #00ffd0 !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }

        /* Tablo altındaki küçük kayıt sayısı yazısı için stil */
        .table-footer {
            font-size: 11px !important;
            color: #8b949e !important;
            margin-top: -8px !important;
            padding-left: 2px;
        }
    </style>
""", unsafe_allow_html=True)

# 2. VERİ YÜKLEME KATMANI
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
data_path = os.path.normpath(os.path.join(project_root, "data", "processed", "inventory_decisions.parquet"))


@st.cache_data
def load_data():
    if not os.path.exists(data_path):
        return None
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    return data


df_raw = load_data()

# 3. BAŞLIK
st.title("📊 OptiChain | Procurement Control Tower")

st.markdown("""
<div class="executive-box">
    <p style="margin:0; font-size:13px; color:#ffffff;">
        <b>Sistem Özeti:</b> Müşteri klik hunisi öncü sinyallerini tarayan <b>LightGBM</b> motoru. 
        Emniyet stokları, yapay zekanın <b>anlık tahmin hatasına (RMSE)</b> ve dinamik servis çarpanına göre kalibre edilmektedir.
    </p>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.error(f"⚠️ HATA: Veri dosyası bulunamadı! Yol: {data_path}")
else:
    # 4. SOL PANEL - KONTROLLER
    st.sidebar.markdown("### 🗓️ Zaman Kontrolü")
    available_dates = sorted(df_raw['date'].unique(), reverse=True)
    selected_date = st.sidebar.selectbox(
        "Planlama Haftası:",
        available_dates,
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )

    st.sidebar.write("---")
    st.sidebar.markdown("### 🛡️ Lojistik Parametreler")

    # Canlı Z-Score Seçimi
    custom_z_high = st.sidebar.slider(
        "Hedef Z-Score (Kritik SKU):",
        min_value=1.00,
        max_value=3.00,
        value=2.33,
        step=0.01,
        help="AX ve AY gibi yüksek ciro döngüsüne sahip ürün gruplarının hedef servis seviyesi çarpanı."
    )

    # Canlı Lead Time Seçimi
    custom_lead_time = st.sidebar.slider(
        "Ortalama Tedarik Süresi (Lead Time - Gün):",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="Sipariş verilen ürünlerin tedarikçiden depoya ulaşması için geçen ortalama gün süresi."
    )

    risk_multiplier = 1.0

    # VERİ FİLTRELEME VE DİNAMİK OPTİMİZASYON MOTORU
    df_current = df_raw[df_raw['date'] == selected_date].copy()

    prev_date_idx = available_dates.index(selected_date) + 1
    if prev_date_idx < len(available_dates):
        df_prev = df_raw[df_raw['date'] == available_dates[prev_date_idx]].copy()
    else:
        df_prev = df_current.copy()

    # Model hatası (RMSE) hesaplama
    df_current['forecast_error'] = df_current['actual_demand'] - df_current['pred_demand']
    df_current['segment_rmse'] = df_current.groupby('abc_xyz_rank', observed=False)['forecast_error'].transform(
        'std').fillna(0.1)

    # --- DİNAMİK LEAD TIME BAĞLANTISI ---
    # Statik kuralları silip doğrudan slider'dan gelen dinamik değeri atıyoruz
    df_current['live_lead_time'] = custom_lead_time

    # Slider'dan gelen canlı Z değerini buraya bağlıyoruz
    df_current['live_z'] = np.where(df_current['abc_xyz_rank'].isin(['AX', 'AY']), custom_z_high, 1.645)

    df_current['sim_safety_stock'] = np.ceil(
        df_current['live_z'] * np.sqrt(df_current['live_lead_time']) * df_current['segment_rmse']).astype(int)

    # Sipariş miktarını lojistik döngüye ve dinamik lead time'a (haftalık bazda) bağlama
    df_current['sim_suggested_qty'] = np.ceil(df_current['pred_demand'] * (custom_lead_time / 7) * risk_multiplier).astype(int)

    # 5. YENİLENEN İŞ DÜNYASI METRİK PANELİ
    st.subheader(f"📈 {selected_date.strftime('%Y-%m-%d')} Operasyon Dönem Karnesi")
    m1, m2, m3, m4, m5 = st.columns(5)

    curr_views = df_current['view_count'].sum()
    prev_views = df_prev['view_count'].sum()
    curr_carts = df_current['cart_count'].sum()
    prev_carts = df_prev['cart_count'].sum()

    this_week_pred = df_current['pred_demand'].sum()
    last_week_actual = df_prev['actual_demand'].sum()
    total_safety = df_current['sim_safety_stock'].sum()
    total_procurement = df_current['sim_suggested_qty'].sum() + total_safety

    m1.metric("👁️ Haftalık Görüntülenme", f"{int(curr_views):,}", f"{int(curr_views - prev_views):+} Trafik")
    m2.metric("🛒 Sepet Aksiyonları", f"{int(curr_carts):,}", f"{int(curr_carts - prev_carts):+} Aksiyon")

    m3.metric(
        label="🔮 Bu Hafta Tahmin",
        value=f"{int(this_week_pred):,} Adet",
        delta=f"Geçen Hafta: {int(last_week_actual):,}",
        delta_color="off"
    )

    m4.metric("🛡️ Toplam Emniyet Stoğu", f"{int(total_safety):,} Adet", f"Z={custom_z_high} Güvencesi",
              delta_color="inverse")
    m5.metric("🚚 Önerilen Satın Alma", f"{int(total_procurement):,} Adet", "Dinamik Tedarik Emri", delta_color="off")

    st.write("---")

    # 6. SEKMELİ MİMARİ
    tab1, tab2 = st.tabs(["📋 SKU Procurement Matrix (Ürün Detayları)", "📊 Executive Insights (Yönetici Özeti)"])

    # TAB 1: ÜRÜN DETAYLARI MATRİSİ
    with tab1:
        st.markdown("### 🔍 Granüler Ürün Sinyalleri ve Satın Alma Emirleri")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            available_options = sorted(df_current['abc_xyz_rank'].dropna().unique())
            safe_defaults = [seg for seg in ['AX', 'AY', 'CZ'] if seg in available_options]
            if not safe_defaults:
                safe_defaults = available_options

            selected_segments = st.multiselect("İzlenecek Envanter Kodları:", options=available_options,
                                               default=safe_defaults)

        with col_f2:
            st.markdown("<br>", unsafe_allow_html=True)
            only_alerts = st.checkbox("🚨 Sadece Riskli veya Kritik Sinyal Veren SKU'ları Listele")

        df_filtered = df_current[df_current['abc_xyz_rank'].isin(selected_segments)]

        if only_alerts:
            df_filtered = df_filtered[df_filtered['pred_demand'] >= (df_filtered['sim_safety_stock'] * 0.1)]

        # Grid görünümü sütun isimleri düzenleme
        df_filtered['final_sku_order'] = df_filtered['sim_suggested_qty'] + df_filtered['sim_safety_stock']

        display_cols = {
            'item_id': 'SKU (Ürün ID)',
            'abc_xyz_rank': 'Envanter Kodu',
            'view_count': 'Haftalık Klik',
            'cart_count': 'Sepet Aksiyonu',
            'actual_demand': 'Gerçekleşen Talep',
            'pred_demand': 'AI Tahmini',
            'sim_safety_stock': 'Emniyet Stoğu (RMSE)',
            'final_sku_order': 'Önerilen Tedarik Miktarı'
        }
        st.dataframe(
            df_filtered[list(display_cols.keys())].rename(columns=display_cols),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f'<p class="table-footer">Filtre kriterlerine uyan toplam listelenen kayıt sayısı: <b>{len(df_filtered):,} SKU</b></p>',
            unsafe_allow_html=True)

    # TAB 2: YÖNETİCİ ÖZETİ VE GRAFİKLER
    with tab2:
        c1, c2 = st.columns([4, 6])
        with c1:
            st.markdown("### 🎯 Seçilen Haftada Segment Dağılımı")
            fig_pie = px.pie(
                df_current,
                names='abc_xyz_rank',
                hole=0.6,
                color_discrete_sequence=['#00ffd0', '#00a2ff', '#7928ca', '#ff007f', '#ffaa00']
            )
            fig_pie.update_layout(
                margin=dict(t=15, b=15, l=10, r=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9')
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("### 📈 Tarihsel Talep Trendi")
            df_trend = df_raw.groupby('date')[['actual_demand', 'pred_demand']].sum().reset_index()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend['date'], y=df_trend['actual_demand'], name="Gerçekleşen Talep",
                                           line=dict(color='#00a2ff', width=3)))
            fig_trend.add_trace(go.Scatter(x=df_trend['date'], y=df_trend['pred_demand'], name="AI Tahmini",
                                           line=dict(color='#00ffd0', width=3, dash='dot')))

            fig_trend.add_vline(x=selected_date.strftime('%Y-%m-%d'), line_dash="dash", line_color="#ff007f")
            fig_trend.update_layout(
                margin=dict(t=15, b=15, l=10, r=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=1.05),
                font=dict(color='#c9d1d9'),
                xaxis=dict(gridcolor='#21262d'),
                yaxis=dict(gridcolor='#21262d')
            )
            st.plotly_chart(fig_trend, use_container_width=True)