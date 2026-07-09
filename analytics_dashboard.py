import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def display_analytics_dashboard(db, symbol="NIFTY50"):
    st.subheader("Market Analytics Dashboard")

    analytics_df = db.get_market_analytics(symbol, days_back=30)

    if not analytics_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=pd.to_datetime(analytics_df['date']),
                y=analytics_df['day_close'],
                mode='lines+markers',
                name='Close Price',
                line=dict(color='#00ff88', width=2)
            ))
            fig_price.update_layout(
                title="30-Day Price Trend",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=pd.to_datetime(analytics_df['date']),
                y=analytics_df['total_volume'],
                name='Volume',
                marker_color='#4444ff'
            ))
            fig_volume.update_layout(
                title="30-Day Volume Trend",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        st.subheader("30-Day Summary")
        metrics = [
            ("Average Price", f"₹{analytics_df['day_close'].mean():,.2f}"),
            ("Volatility (σ)", f"{analytics_df['price_change_pct'].std():.2f}%"),
            ("Max Daily Gain", f"{analytics_df['price_change_pct'].max():.2f}%"),
            ("Max Daily Loss", f"{analytics_df['price_change_pct'].min():.2f}%")
        ]
        for col, (label, val) in zip(st.columns(4), metrics):
            col.metric(label, val)
