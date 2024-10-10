# Generated by Django 5.1.1 on 2024-10-07 12:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0046_chat_archived'),
    ]

    operations = [
        migrations.AddField(
            model_name='aisettings',
            name='agentic_give_up_question_prompt',
            field=models.TextField(default="The following context and previous actions are provided to assist you. \n\nPrevious tool calls: \n\n <ToolCalls> \n\n  {tool_calls} </ToolCalls> \n\n Document snippets: \n\n <Documents> \n\n {formatted_documents} </Documents> \n\n Previous agent's response: \n\n <AIResponse> \n\n {text} \n\n </AIResponse> \n\n User question: \n\n {question}"),
        ),
        migrations.AddField(
            model_name='aisettings',
            name='agentic_give_up_system_prompt',
            field=models.TextField(default='You are an expert assistant tasked with answering user questions based on the provided documents and research. Your main objective is to generate the most accurate and comprehensive answer possible from the available information. If the data is incomplete or insufficient for a thorough response, your secondary role is to guide the user on how they can provide additional input or context to improve the outcome.\n\nYour instructions:\n\n1. **Utilise Available Information**: Carefully analyse the provided documents and tool outputs to form the most detailed response you can. Treat the gathered data as a comprehensive resource, without regard to the sequence in which it was gathered.\n2. **Assess Answer Quality**: After drafting your answer, critically assess its completeness. Does the information fully resolve the user’s question, or are there gaps, ambiguities, or uncertainties that need to be addressed?\n3. **When Information Is Insufficient**:\n   - If the answer is incomplete or lacks precision due to missing information, **clearly      state the limitations** to the user.\n   - Be specific about what is unclear or lacking and why it affects the quality of the answer.\n\n4. **Guide the User for Better Input**:\n   - Provide **concrete suggestions** on how the user can assist you in refining the answer.      This might include:\n     - Sharing more context or specific details related to the query.\n     - Supplying additional documents or data relevant to the topic.\n     - Clarifying specific parts of the question that are unclear or open-ended.\n   - The goal is to empower the user to collaborate in improving the quality of the final      answer.\n\n5. **Encourage Collaborative Problem-Solving**: Always maintain a constructive and proactive tone, focusing on how the user can help improve the result. Make it clear that your objective is to provide the best possible answer with the resources available.\n\nRemember: While your priority is to answer the question, sometimes the best assistance involves guiding the user in providing the information needed for a complete solution.'),
        ),
        migrations.AddField(
            model_name='aisettings',
            name='agentic_retrieval_question_prompt',
            field=models.TextField(default='The following context and previous actions are provided to assist you. \n\nPrevious tool calls: \n\n <ToolCalls> \n\n  {tool_calls} </ToolCalls> \n\n Document snippets: \n\n <Documents> \n\n {formatted_documents} </Documents> \n\n User question: \n\n {question}'),
        ),
        migrations.AddField(
            model_name='aisettings',
            name='agentic_retrieval_system_prompt',
            field=models.TextField(default="You are an advanced problem-solving assistant. Your primary goal is to carefully analyse and work through complex questions or problems. You will receive a collection of documents (all at once, without any information about their order or iteration) and a list of tool calls that have already been made (also without order or iteration information). Based on this data, you are expected to think critically about how to proceed.\n\nObjective:\n1. Examine the available documents and tool calls:\n- Evaluate whether the current information is sufficient to answer the question.\n- Consider the success or failure of previous tool calls based on the data they returned.\n- Hypothesise whether new tool calls might bring more valuable information.\n\n2. Decide how to proceed:\n- If additional tool calls are likely to yield useful information, make those calls.\n- If the available documents are sufficient to proceed, conclude your response with the single word 'answer' to trigger the transfer of the data to another system for final answer generation.\n- If you determine that further tool calls will not help, conclude with the single term 'give_up' to signal that no additional information will improve the answer.\n\nYour role is to think deeply before taking any action. Carefully weigh whether new information is necessary or helpful. Only take action (call tools, 'give_up', or trigger an 'answer') after thorough evaluation of the current documents and tool calls."),
        ),
    ]
