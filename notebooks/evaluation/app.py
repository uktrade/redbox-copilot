from analysis_of_chat_history import ChatHistoryAnalysis
import streamlit as st

st.set_page_config(page_title = 'Redbox Chat Analysis', layout = 'wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Redbox Chat History Dashboard')

cha = ChatHistoryAnalysis()

user_freq_tab, traffic_tab, word_freq_tab, ai_word_freq_tab = st.tabs(['User Frequency',
                                                                       'Redbox traffic',
                                                                       'Word frequency',
                                                                       'AI word frequency',
                                                                       'AI response pattern'])

with user_freq_tab:
    st.pyplot(cha.user_frequency_analysis())

with traffic_tab:
    st.pyplot(cha.redbox_traffic_analysis())

with word_freq_tab:
    st.pyplot(cha.user_word_frequency_analysis())

with ai_word_freq_tab:
    st.pyplot(cha.ai_word_frequency_analysis())

# with ai_response_tab:
#     st.pyplot(cha.ai_response_pattern_analysis())