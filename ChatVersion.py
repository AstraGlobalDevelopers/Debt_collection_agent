from Assistant import MasterDebtCollectorAssistant

assistant = MasterDebtCollectorAssistant()

while True:
    user_input = input("You: ")
    assistant_dict_response, transition = assistant.get_assistant_response(user_input)
    # assistant_dict_response = json.loads(assistant_json_response)
    collector_response = assistant_dict_response.get("Debt_Collector_Response")

    print(f"Debt Collector: {collector_response}")