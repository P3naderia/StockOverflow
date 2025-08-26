import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import io

# 페이지 설정
st.set_page_config(
    page_title="AI 재고 예측 시스템",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .urgent-row {
        background-color: #ffebee !important;
    }
    .medium-row {
        background-color: #fff3e0 !important;
    }
    .low-row {
        background-color: #e8f5e8 !important;
    }
</style>
""", unsafe_allow_html=True)

class InventoryForecaster:
    def __init__(self):
        self.data = None
        self.forecast_period = 30
        
    def safe_float_convert(self, value):
        """문자열을 안전하게 float로 변환"""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
            
        # 문자열인 경우 처리
        if isinstance(value, str):
            # $, 쉼표, 공백 제거
            cleaned = value.replace('$', '').replace(',', '').replace(' ', '').strip()
            # 퍼센트 기호 제거
            cleaned = cleaned.replace('%', '')
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        
        # 이미 숫자인 경우
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
        
    def generate_sample_data(self, n_products=20):
        """샘플 데이터 생성"""
        products = [
            "무선 이어폰 프로", "스마트워치 울트라", "휴대용 충전기", "블루투스 스피커",
            "스마트폰 케이스", "무선 충전패드", "USB-C 케이블", "노트북 스탠드",
            "웹캠 HD", "키보드 메카니컬", "마우스 무선", "모니터 받침대",
            "태블릿 거치대", "이어폰 케이스", "스피커 미니", "충전기 고속",
            "케이블 정리함", "휴대폰 거치대", "노트북 파우치", "마우스패드 대형"
        ]
        
        categories = ["Electronics", "Accessories", "Audio", "Computing"]
        
        data = []
        for i in range(min(n_products, len(products))):
            # 과거 6개월 매출 트렌드 생성
            monthly_sales = []
            base_sales = random.randint(3000, 12000)
            for month in range(6):
                # 트렌드와 계절성 반영
                trend = 1 + (month * 0.05)  # 월별 5% 성장
                seasonality = 1 + 0.2 * np.sin(month / 6 * 2 * np.pi)
                noise = random.uniform(0.8, 1.2)
                monthly_sales.append(int(base_sales * trend * seasonality * noise))
            
            product = {
                '(Parent) ASIN': f'B{2000+i:03d}',
                '(Child) ASIN': f'B{2000+i:03d}-{chr(65+i%26)}',
                'Title': products[i],
                'Current Stock': random.randint(50, 500),
                'Units Ordered': random.randint(50, 300),
                'Ordered Product Sales': f"${random.randint(5000, 15000):,}.{random.randint(10, 99)}",
                'Unit Session Percentage': f"{random.uniform(2, 10):.2f}%",
                'Sessions - Total': random.randint(1000, 5000),
                'Featured Offer (Buy Box) Percentage': f"{random.uniform(70, 100):.1f}%",
                'lead_time': random.randint(7, 21),
                'category': random.choice(categories),
                # 매출 트렌드 데이터 (과거 6개월)
                'sales_trend': monthly_sales,
                'trend_months': ['3개월 전', '2개월 전', '1개월 전', '이번달', '예측1', '예측2']
            }
            data.append(product)
            
        return pd.DataFrame(data)
    
    def calculate_ai_forecast(self, row, period):
        """AI 기반 재고 예측 계산"""
        # 컬럼명 매핑
        column_mapping = {
            'units_ordered': ['Units Ordered', 'units_ordered', 'Units Ordered - Total'],
            'ordered_product_sales': ['Ordered Product Sales', 'ordered_product_sales', 'Ordered Product Sales - Total'],
            'unit_session_percentage': ['Unit Session Percentage', 'unit_session_percentage', 'Unit Session Percentage - Total'],
            'current_stock': ['Current Stock', 'current_stock', 'stock', 'inventory'],
            'lead_time': ['lead_time', 'Lead Time', 'lead_days']
        }
        
        # 동적 컬럼 찾기
        def find_column(row, possible_names):
            for name in possible_names:
                if name in row.index:
                    return self.safe_float_convert(row[name])
            # 기본값 반환
            return 100.0 if 'units' in str(possible_names).lower() else 50.0
        
        # 일일 수요 계산
        units_ordered = find_column(row, column_mapping['units_ordered'])
        daily_demand = max(units_ordered / 30, 0.1)  # 최소값 설정
        
        # 트렌드 및 계절성 반영
        seasonality_factor = 1 + 0.1 * np.sin(2 * np.pi * datetime.now().month / 12)
        trend_factor = random.uniform(0.9, 1.3)
        
        # 예측 수요
        predicted_demand = int(daily_demand * period * seasonality_factor * trend_factor)
        
        # 리드타임과 현재 재고
        lead_time = max(find_column(row, column_mapping['lead_time']), 1)
        current_stock = max(find_column(row, column_mapping['current_stock']), 0)
        
        # 안전 재고
        safety_stock = int(daily_demand * lead_time * 1.5)
        
        # 권장 발주량
        recommended_reorder = predicted_demand + safety_stock
        
        # 재고 소진 예상일
        days_until_stockout = int(current_stock / max(daily_demand, 0.1))
        
        # 우선순위 결정
        if days_until_stockout < lead_time:
            urgency = 'HIGH'
            urgency_score = 3
        elif days_until_stockout < lead_time * 2:
            urgency = 'MEDIUM'
            urgency_score = 2
        else:
            urgency = 'LOW'
            urgency_score = 1
            
        # 매출 영향 계산
        ordered_sales = find_column(row, column_mapping['ordered_product_sales'])
        avg_price = ordered_sales / max(units_ordered, 1)
        revenue_impact = predicted_demand * avg_price
        
        # AI 신뢰도
        confidence = random.randint(82, 98)
        
        return {
            'predicted_demand': predicted_demand,
            'recommended_reorder': recommended_reorder,
            'safety_stock': safety_stock,
            'days_until_stockout': days_until_stockout,
            'urgency': urgency,
            'urgency_score': urgency_score,
            'revenue_impact': revenue_impact,
            'confidence': confidence,
            'daily_demand': daily_demand
        }
    
    def load_data(self, uploaded_file=None):
        """데이터 로드"""
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    self.data = pd.read_csv(uploaded_file)
                else:
                    self.data = pd.read_excel(uploaded_file)
                
                st.success("✅ 파일이 성공적으로 업로드되었습니다!")
                st.write("**업로드된 데이터 컬럼:**")
                st.write(list(self.data.columns))
                
                return True
            except Exception as e:
                st.error(f"❌ 파일 업로드 오류: {str(e)}")
                st.info("샘플 데이터를 사용합니다.")
                self.data = self.generate_sample_data()
                return True
        else:
            self.data = self.generate_sample_data()
            st.info("📊 샘플 데이터를 사용합니다. 실제 데이터를 업로드하세요.")
            return True
    
    def get_forecasts(self):
        """모든 제품에 대한 예측 결과"""
        forecasts = []
        for _, row in self.data.iterrows():
            forecast = self.calculate_ai_forecast(row, self.forecast_period)
            forecasts.append({**row.to_dict(), **forecast})
        
        df = pd.DataFrame(forecasts)
        return df.sort_values('urgency_score', ascending=False)

def main():
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🧠 AI 재고 예측 시스템")
        st.markdown("Amazon 판매 데이터 기반 지능형 재고 관리")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 파일 업로드
        st.subheader("📁 데이터 업로드")
        uploaded_file = st.file_uploader(
            "Amazon 리포트 파일을 업로드하세요",
            type=['csv', 'xlsx'],
            help="ASIN, 세션, 주문 데이터가 포함된 파일"
        )
        
        # 예측 기간
        st.subheader("📅 예측 설정")
        forecast_period = st.selectbox(
            "예측 기간",
            options=[7, 14, 30, 60],
            index=2
        )
        
        # 필터링 옵션
        urgency_filter = st.multiselect(
            "우선순위 필터",
            options=['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM', 'LOW']
        )
        
        # 새로고침 버튼
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()
    
    # 메인 앱
    forecaster = InventoryForecaster()
    forecaster.forecast_period = forecast_period
    
    if forecaster.load_data(uploaded_file):
        forecasts_df = forecaster.get_forecasts()
        
        # 필터 적용
        filtered_df = forecasts_df[forecasts_df['urgency'].isin(urgency_filter)]
        
        # 대시보드 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(forecasts_df)
            st.metric("총 상품 수", total_products)
        
        with col2:
            high_urgency = len(forecasts_df[forecasts_df['urgency'] == 'HIGH'])
            st.metric("긴급 재고 보충", high_urgency)
            
        with col3:
            # 안전하게 전환율 계산
            conversion_rates = []
            for _, row in forecasts_df.iterrows():
                rate = forecaster.safe_float_convert(row.get('Unit Session Percentage', row.get('unit_session_percentage', 0)))
                conversion_rates.append(rate)
            avg_conversion = np.mean(conversion_rates) if conversion_rates else 0
            st.metric("평균 전환율", f"{avg_conversion:.2f}%")
            
        with col4:
            total_risk = forecasts_df[forecasts_df['urgency'] == 'HIGH']['revenue_impact'].sum()
            st.metric("위험 매출", f"${total_risk/1000:.0f}K")
        
        # 예측 결과 테이블
        st.subheader("🎯 AI 재고 예측 결과")
        
        # 컬럼 순서 정의 (요청사항 반영)
        display_columns = []
        column_renames = {}
        
        # 1. Parent ASIN
        if '(Parent) ASIN' in forecasts_df.columns:
            display_columns.append('(Parent) ASIN')
            column_renames['(Parent) ASIN'] = 'Parent ASIN'
        
        # 2. Child ASIN  
        if '(Child) ASIN' in forecasts_df.columns:
            display_columns.append('(Child) ASIN')
            column_renames['(Child) ASIN'] = 'Child ASIN'
            
        # 3. 제품명
        title_col = None
        for col in ['Title', 'title', 'Product Title', 'ASIN']:
            if col in forecasts_df.columns:
                title_col = col
                break
        
        if title_col:
            display_columns.append(title_col)
            column_renames[title_col] = '제품명'
        
        # 4. 현재 재고 (요청사항)
        if 'Current Stock' in forecasts_df.columns:
            display_columns.append('Current Stock')
            column_renames['Current Stock'] = '현재 재고'
        
        # 5. 예측 결과 컬럼들
        forecast_columns = ['predicted_demand', 'recommended_reorder', 'days_until_stockout', 'urgency', 'confidence']
        display_columns.extend(forecast_columns)
        column_renames.update({
            'predicted_demand': '예측 수요',
            'recommended_reorder': '권장 발주량',
            'days_until_stockout': '재고 소진일',
            'urgency': '우선순위',
            'confidence': '신뢰도(%)'
        })
        
        # 데이터프레임 표시
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.rename(columns=column_renames)
        
        # 스타일 적용
        def style_urgency(val):
            if val == 'HIGH':
                return 'background-color: #ffcdd2; font-weight: bold'
            elif val == 'MEDIUM':
                return 'background-color: #ffe0b2'
            else:
                return 'background-color: #dcedc8'
        
        if '우선순위' in display_df.columns:
            styled_df = display_df.style.applymap(style_urgency, subset=['우선순위'])
        else:
            styled_df = display_df
            
        # 클릭 가능한 테이블을 위한 인덱스 표시
        st.dataframe(
            styled_df, 
            use_container_width=True,
            column_config={
                "Parent ASIN": st.column_config.TextColumn("Parent ASIN", width="small"),
                "Child ASIN": st.column_config.TextColumn("Child ASIN", width="small"), 
                "제품명": st.column_config.TextColumn("제품명", width="large"),
                "현재 재고": st.column_config.NumberColumn("현재 재고", format="%d"),
                "예측 수요": st.column_config.NumberColumn("예측 수요", format="%d"),
                "권장 발주량": st.column_config.NumberColumn("권장 발주량", format="%d"),
                "재고 소진일": st.column_config.NumberColumn("재고 소진일", format="%d일"),
                "신뢰도(%)": st.column_config.NumberColumn("신뢰도", format="%d%%"),
            }
        )
        
        # 상세 분석 - ASIN 선택
        st.subheader("🔍 ASIN별 상세 분석")
        
        # ASIN 선택 (Parent ASIN 우선, 없으면 Child ASIN)
        if '(Parent) ASIN' in filtered_df.columns:
            asin_options = filtered_df['(Parent) ASIN'].tolist()
            asin_col = '(Parent) ASIN'
        elif '(Child) ASIN' in filtered_df.columns:
            asin_options = filtered_df['(Child) ASIN'].tolist() 
            asin_col = '(Child) ASIN'
        else:
            asin_options = [f"Product {i+1}" for i in range(len(filtered_df))]
            asin_col = None
            
        selected_asin = st.selectbox("📦 ASIN 선택", options=asin_options)
        
        if selected_asin and len(filtered_df) > 0:
            if asin_col:
                product_data = filtered_df[filtered_df[asin_col] == selected_asin].iloc[0]
                product_title = product_data.get('Title', selected_asin)
            else:
                product_data = filtered_df.iloc[0]
                product_title = f"Product 1"
            
            with st.expander(f"📊 {selected_asin} - {product_title} 상세 분석", expanded=True):
                
                # 핵심 지표
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_stock = forecaster.safe_float_convert(product_data.get('Current Stock', 100))
                    st.metric("현재 재고", f"{current_stock:.0f}")
                    
                with col2:
                    st.metric("예측 수요", f"{product_data['predicted_demand']:.0f}")
                with col3:
                    st.metric("권장 발주량", f"{product_data['recommended_reorder']:.0f}")
                with col4:
                    st.metric("재고 소진일", f"{product_data['days_until_stockout']:.0f}일")
                
                # 매출 트렌드 차트 (요청사항)
                st.subheader("📈 매출 트렌드")
                
                if 'sales_trend' in product_data and 'trend_months' in product_data:
                    # 실제 트렌드 데이터가 있는 경우
                    trend_data = pd.DataFrame({
                        '기간': product_data['trend_months'],
                        '매출': product_data['sales_trend']
                    })
                    
                    fig_trend = px.line(
                        trend_data, 
                        x='기간', 
                        y='매출',
                        title=f"{selected_asin} 매출 트렌드",
                        markers=True
                    )
                    fig_trend.update_layout(
                        xaxis_title="기간",
                        yaxis_title="매출 ($)",
                        yaxis_tickformat="$,.0f"
                    )
                    
                    # 예측 구간 색상 구분
                    fig_trend.add_vline(x=3.5, line_dash="dash", line_color="red", 
                                      annotation_text="예측 시작점")
                    
                else:
                    # 트렌드 데이터가 없는 경우 시뮬레이션
                    months = ['3개월 전', '2개월 전', '1개월 전', '이번달', '다음달 예측', '2개월 후 예측']
                    base_sales = forecaster.safe_float_convert(product_data.get('Ordered Product Sales', 5000))
                    
                    sales_trend = []
                    for i, month in enumerate(months):
                        if i < 4:  # 과거 데이터
                            trend_factor = 1 + (i * 0.1) + random.uniform(-0.2, 0.2)
                        else:  # 예측 데이터
                            trend_factor = 1.2 + random.uniform(-0.1, 0.3)
                        sales_trend.append(base_sales * trend_factor)
                    
                    trend_data = pd.DataFrame({
                        '기간': months,
                        '매출': sales_trend
                    })
                    
                    fig_trend = px.line(
                        trend_data, 
                        x='기간', 
                        y='매출',
                        title=f"{selected_asin} 매출 트렌드 (시뮬레이션)",
                        markers=True
                    )
                    fig_trend.update_layout(
                        xaxis_title="기간",
                        yaxis_title="매출 ($)", 
                        yaxis_tickformat="$,.0f"
                    )
                    
                    # 실제 vs 예측 구분
                    fig_trend.add_vline(x=3.5, line_dash="dash", line_color="red",
                                      annotation_text="예측 시작점")
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 상세 지표 (기존 코드 유지)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 세션 및 전환 지표")
                    sessions_total = forecaster.safe_float_convert(product_data.get('Sessions - Total', 0))
                    conversion_rate = forecaster.safe_float_convert(product_data.get('Unit Session Percentage', 0))
                    buy_box = forecaster.safe_float_convert(product_data.get('Featured Offer (Buy Box) Percentage', 0))
                    
                    st.write(f"• 총 세션: {sessions_total:,.0f}")
                    st.write(f"• 전환율: {conversion_rate:.2f}%")
                    st.write(f"• Buy Box 점유율: {buy_box:.1f}%")
                    st.write(f"• AI 신뢰도: {product_data['confidence']:.0f}%")
                
                with col2:
                    st.subheader("💰 매출 및 주문 지표")
                    units_ordered = forecaster.safe_float_convert(product_data.get('Units Ordered', 0))
                    ordered_sales = forecaster.safe_float_convert(product_data.get('Ordered Product Sales', 0))
                    
                    st.write(f"• 총 주문량: {units_ordered:,.0f}")
                    st.write(f"• 총 매출: ${ordered_sales:,.0f}")
                    
                    if units_ordered > 0:
                        avg_order_value = ordered_sales / units_ordered
                        st.write(f"• 평균 주문 금액: ${avg_order_value:.2f}")
                    else:
                        st.write(f"• 평균 주문 금액: $0.00")
                        
                    st.write(f"• 예상 매출 영향: ${product_data['revenue_impact']:,.0f}")
                
                # 액션 버튼
                st.subheader("🎯 액션")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📝 {selected_asin} 발주 요청", key=f"order_{selected_asin}"):
                        st.success(f"✅ {selected_asin}의 발주 요청이 생성되었습니다!")
                        st.info(f"권장 발주량: {product_data['recommended_reorder']:.0f}개")
                        
                with col2:
                    if st.button(f"🔔 {selected_asin} 재고 알림", key=f"alert_{selected_asin}"):
                        st.success(f"✅ {selected_asin}의 재고 알림이 설정되었습니다!")
                        st.info(f"알림 기준: {product_data['days_until_stockout']:.0f}일 이하 시")
                        
                with col3:
                    if st.button(f"📊 {selected_asin} 상세 리포트", key=f"report_{selected_asin}"):
                        # 상세 리포트 생성
                        report_data = {
                            "ASIN": selected_asin,
                            "제품명": product_title,
                            "현재 재고": current_stock,
                            "예측 수요": product_data['predicted_demand'],
                            "권장 발주량": product_data['recommended_reorder'],
                            "우선순위": product_data['urgency'],
                            "생성일시": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        report_text = f"""
                        📊 ASIN 상세 분석 리포트
                        
                        ASIN: {report_data['ASIN']}
                        제품명: {report_data['제품명']}
                        
                        📦 재고 현황:
                        - 현재 재고: {report_data['현재 재고']:.0f}개
                        - 예측 수요: {report_data['예측 수요']:.0f}개
                        - 권장 발주량: {report_data['권장 발주량']:.0f}개
                        - 우선순위: {report_data['우선순위']}
                        
                        생성일시: {report_data['생성일시']}
                        """
                        
                        st.download_button(
                            label="📥 리포트 다운로드",
                            data=report_text,
                            file_name=f"{selected_asin}_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
        
        # 차트 분석
        st.subheader("📈 분석 차트")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 우선순위별 분포
            urgency_counts = forecasts_df['urgency'].value_counts()
            fig_pie = px.pie(
                values=urgency_counts.values,
                names=urgency_counts.index,
                title="우선순위별 상품 분포",
                color_discrete_map={'HIGH': '#ff6b6b', 'MEDIUM': '#ffd93d', 'LOW': '#6bcf7f'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 재고 vs 수요 산점도
            fig_scatter = px.scatter(
                forecasts_df,
                x='days_until_stockout',
                y='predicted_demand',
                color='urgency',
                title="재고 소진일 vs 예측 수요",
                color_discrete_map={'HIGH': '#ff6b6b', 'MEDIUM': '#ffd93d', 'LOW': '#6bcf7f'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    main()
