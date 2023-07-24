import streamlit as st
import pickle
import os
import time
import json
from streamlit import session_state

os.environ['http_proxy'] = 'http://127.0.0.1:8080'
os.environ["https_proxy"] = "http://127.0.0.1:8080"
# streamlit run dataset.py --server.port 2323
st.set_page_config(
    page_title='城理问答数据集生成器',
    layout="wide",
    page_icon='😅',
    initial_sidebar_state="expanded",  # “auto”或“expanded”或“collapsed”
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


# 加载问题库
def load_questions(file_path):
    if not os.path.exists(file_path):
        st.error(f"文件 {file_path} 不存在")
        print(f"文件 {file_path} 不存在")
        return {}
    else:
        with open(file_path, "r", encoding='utf-8') as file:
            questions = file.readlines()
        return list(set([q.strip() for q in questions if q != '' and q != '\n']))  # 去重


# 保存问题库
def save_questions(file_path, questions):
    with open(file_path, "w", encoding='utf-8') as file:
        for question in questions:
            file.write(question + "\n")


# 保存临时回答
def save_answers(temp_answers, just_read=False):
    if just_read:
        if os.path.exists("data.pkl"):
            with open("data.pkl", "rb") as file:
                answers = pickle.load(file)
        else:
            answers = {}
        session_state.all_answers = answers
        return True
    else:
        if os.path.exists("data.pkl"):
            with open("data.pkl", "rb") as file:
                answers = pickle.load(file)
        answers.update(temp_answers)  # 覆盖式更新
        with open("data.pkl", "wb") as file:
            pickle.dump(answers, file)
        session_state.all_answers = answers
        return True

# def save_answers_as_json(answers, file_path):
#     history = []
#     if answers == '':
#         pass
#     else:
#         for question, answer in answers.items():
#             history.append([question, answer])
#         if len(history) == 1:
#             history=[]
#         else:
#             history = history[:-1]
#         item = {"prompt": question, "response": answer, "history": history}
#         with open(file_path, "a", encoding="utf-8") as file:
#             json.dump(item, file, ensure_ascii=False, indent=2)
def save_answers_as_json(answers, file_path):
    if answers == '':
        pass
    else:
        for content, summary in answers.items():
            item = {"content": content, "summary": summary}
        with open(file_path, "a", encoding="utf-8") as file:
            json.dump(item, file, ensure_ascii=False, indent=2)

def reset_text_area():
    if session_state.text_area_tittle == "回答：(内容为空则不保存此回答)":
        session_state.text_area_tittle = "回答：(内容为空则不保存此回答) "
    elif session_state.text_area_tittle == "回答：(内容为空则不保存此回答) ":
        session_state.text_area_tittle = "回答：(内容为空则不保存此回答)"


def main():
    st.title("城理问答数据集生成器")
    if 'temp_answers' not in session_state:
        session_state.temp_answers = {}
    if 'all_answers' not in session_state:
        save_answers(session_state.temp_answers, just_read=True)
        session_state.question_txt = "questions.txt"
        session_state.answers_json = "answers.json"
        session_state.generated_answer = ""
        session_state.text_area_tittle = "回答：(内容为空则不保存此回答)"
        session_state.selected_id = 0
    session_state.question_txt = st.sidebar.text_input("存有每一条问题的txt", value=session_state.question_txt)
    session_state.answers_json = st.sidebar.text_input("保存回答的json路径", value=session_state.answers_json)
    if 'questions' not in session_state:
        session_state.questions = load_questions(session_state.question_txt)
    selected_questions = {}
    for q in range(len(session_state.questions)):
        selected_questions[session_state.questions[q]] = q
    selectbox_empty = st.empty()
    selected_question = selectbox_empty.selectbox("请选择一个问题：", session_state.questions,
                                                  index=session_state.selected_id)
    if selected_question:
        session_state.selected_id = selected_questions[selected_question]
        selected_question = selectbox_empty.selectbox("请选择一个问题： ", session_state.questions,
                                                      index=session_state.selected_id)
        user_answer_empty = st.empty()
        user_answer = user_answer_empty.text_area(session_state.text_area_tittle, session_state.generated_answer,
                                                  height=200)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("保存全部回答(未保存超过10个会自动保存的)") or len(session_state.temp_answers) >= 10:
                session_state.selected_id = 0
                # for question in session_state.temp_answers:  # 删除已经回答的问题，但可以覆盖data.pkl存过的问题
                #     session_state.questions.remove(question)

                #
                save_questions(session_state.question_txt, session_state.questions)

                if save_answers(session_state.temp_answers):
                    st.success("全部回答已保存。")
                    session_state.temp_answers = {}
                else:
                    st.error("保存失败，请稍后重试。经常出现此问题是因为死锁，请删除data.pkl文件后重试。")
                    time.sleep(5)
                st.experimental_rerun()

        with col2:
            if st.button("确认此回答(自动下一个)"):
                session_state.generated_answer = ''
                if user_answer != '':
                    session_state.temp_answers[selected_question] = user_answer
                elif selected_question in session_state.temp_answers:  # 内容为空则不保存此回答
                    del session_state.temp_answers[selected_question]  # data.pkl存过的问题不清空
                reset_text_area()
                user_answer = user_answer_empty.text_area(session_state.text_area_tittle, height=200)
                session_state.selected_id += 1
                if session_state.selected_id >= len(session_state.questions):
                    session_state.selected_id = 0
                st.experimental_rerun()
        with col3:
            if st.button("上一个问题"):
                session_state.generated_answer = ''
                session_state.selected_id -= 1
                if session_state.selected_id < 0:
                    session_state.selected_id = len(session_state.questions) - 1
                reset_text_area()
                user_answer = user_answer_empty.text_area(session_state.text_area_tittle, height=200)
                st.experimental_rerun()
        with col4:
            if st.button("下一个问题"):
                session_state.generated_answer = ''
                session_state.selected_id += 1
                if session_state.selected_id >= len(session_state.questions):
                    session_state.selected_id = 0
                reset_text_area()
                user_answer = user_answer_empty.text_area(session_state.text_area_tittle, height=200)
                st.experimental_rerun()
    if st.sidebar.button("清除复位"):
        session_state.all_answers = {}
        with open('./data.pkl','wb')as f:
            pickle.dump({},f)
            f.close()
    if st.sidebar.button("导出载入的已保存回答为 JSON"):
        save_answers_as_json(session_state.all_answers, session_state.answers_json)
    st.json({"未保存回答：": session_state.temp_answers, "已保存回答：": session_state.all_answers})


if __name__ == "__main__":
    main()
