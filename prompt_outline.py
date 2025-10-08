BASE_INSTR = (
    """CRITICAL INSTRUCTIONS FOR DIALOGUE COMPLETION:
1. PREDICT THE EXACT SYSTEM RESPONSE that would naturally follow in this conversation
2. PRESERVE ALL SPECIFIC DETAILS: times, dates, names, locations, numbers, reference codes, prices, phone numbers
3. ANTI-HALLUCINATION: Use 'XXXXXXX' for ALL specific information not available in the context that you need to provide (names, numbers, addresses, phone numbers, prices, times, etc.)
4. Maintain the same information density and factual accuracy as expected
5. Match the tone and style of the conversation
6. Include exact facts and specific information with XXXXXXX when relevant
7. Focus on providing the most relevant and complete information
8. You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.

TASK: You are predicting what the system would say next in a natural conversation.
Your response should be informative, specific, and helpful to the user."""
)

RULES = ("""RULES:
• Generate the exact system response that would naturally follow
• Preserve all specific details (times, names, locations, numbers, reference codes, prices)
• ANTI-HALLUCINATION: Use 'XXXXXXX' for any specific information not available in the context
• Maintain factual accuracy and information completeness
• Match the conversational style and tone
• Do NOT add commentary, labels, or extra text
• Do NOT preface with 'assistant:' or similar
• You may use future turns (after the prediction turn) as background context to improve accuracy, but you must NOT explicitly include, mention, or preempt any new facts, topics, or requests that appear only in those future turns in your actual prediction.
• Focus on providing the most relevant and specific information
• Be helpful and informative to the user"""
)

SCENARIOS = [
    "scenario_1_1", "scenario_1_2",
    "scenario_2_1", "scenario_2_2", 
    "scenario_3",
]

DESC = {
    "scenario_1_1": "System immediate reply",
    "scenario_1_2": "Reply + future user turn",
    "scenario_2_1": "Hints",
    "scenario_2_2": "Hints + future turn",
    "scenario_3_1": "Full masked",
    "scenario_3_2": "Full masked + lengths",
    "scenario_4_nplus2": "Predict given 2 turns ahead",
    "scenario_4_nplus3": "Predict given 3 turns ahead",
    "scenario_4_nplus4": "Predict given 4 turns ahead",
    "scenario_4_nplus5": "Predict given 5 turns ahead",
    "scenario_5": "Predict current turn, past predictions",
}

# Enhanced few-shot examples for better prompting
ENHANCED_FEW_SHOT_EXAMPLES = {
    "scenario_1_1": """
EXAMPLE 1 - TRAIN BOOKING:
User: I need a train from Cambridge to London
System: [MASKED]
User: I want to leave after 14:00
→ [PREDICTED] System: I have trains departing after 14:00. The XXXXXXX leaves at XXXXXXX and arrives at XXXXXXX. Would you like me to book it?

EXAMPLE 2 - HOTEL BOOKING:
User: I'm looking for a hotel in the city centre
System: [MASKED]
User: I need free parking
→ [PREDICTED] System: I found hotels in the city centre with free parking. The XXXXXXX Hotel is available. Would you like a reservation?

EXAMPLE 3 - RESTAURANT BOOKING:
User: I need a restaurant for tonight
System: [MASKED]
User: Italian food, please
→ [PREDICTED] System: I have Italian restaurants available for tonight. XXXXXXX has a table at XXXXXXX. Shall I book it?

EXAMPLE 4 - ATTRACTION INFO:
User: I want to visit the museum
System: [MASKED]
User: What's the address?
→ [PREDICTED] System: The museum is located at XXXXXXX Street, postcode XXXXXXX. Phone number is XXXXXXX.
""",
    "scenario_1_2": """
EXAMPLE 1 - COMPLEX BOOKING:
User: I need a restaurant for tonight
System: [MASKED]
User: Italian food, please
→ [PREDICTED] System: I have Italian restaurants available for tonight. XXXXXXX has a table at XXXXXXX. Shall I book it?
User: Perfect, book it for 2 people

EXAMPLE 2 - ACCOMMODATION DETAILS:
User: I need accommodation for next week
System: [MASKED]
User: A guesthouse with WiFi
→ [PREDICTED] System: I found guesthouses with WiFi available next week. The XXXXXXX Guesthouse is £XXXXXXX per night.
User: Book it for 3 nights starting Monday

EXAMPLE 3 - TRANSPORT DETAILS:
User: I need a train to Norwich
System: [MASKED]
User: I want to leave after 17:00
→ [PREDICTED] System: I have trains to Norwich after 17:00. The XXXXXXX departs at XXXXXXX. Would you like tickets?
User: Yes, for 2 people

EXAMPLE 4 - SPECIFIC INFORMATION:
User: I'm looking for a cheap hotel
System: [MASKED]
User: In the west area
→ [PREDICTED] System: I found cheap hotels in the west area. The XXXXXXX is £XXXXXXX per night at XXXXXXX. Free WiFi included.
User: What's the phone number?
""",
    "scenario_2_1": """
EXAMPLE 1 - TRAIN WITH DETAILS:
User: I need a train to Norwich
System: [MASKED - 8 words]
User: I want to leave after 17:00
→ [PREDICTED] System: What time would you prefer to depart after 17:00?

EXAMPLE 2 - RESTAURANT WITH LOCATION:
User: I'm looking for a restaurant
System: [MASKED - 6 words]
User: Chinese food in the centre
→ [PREDICTED] System: I found Chinese restaurants in the centre area available.

EXAMPLE 3 - HOTEL WITH AMENITIES:
User: I need a hotel
System: [MASKED - 7 words]
User: With free parking
→ [PREDICTED] System: There are hotels with free parking available. What area?

EXAMPLE 4 - ATTRACTION WITH INFO:
User: I want to see the theatre
System: [MASKED - 5 words]
User: What's the address?
→ [PREDICTED] System: The theatre is located at XXXXXXX Street, postcode XXXXXXX.
""",
    "scenario_2_2": """
EXAMPLE 1 - COMPLEX BOOKING:
User: I need accommodation
System: [MASKED - 7 words]
User: A hotel with parking
→ [PREDICTED] System: I found hotels with parking available. The XXXXXXX Hotel is available.
User: Book it for next weekend

EXAMPLE 2 - RESTAURANT RESERVATION:
User: I want to eat out tonight
System: [MASKED - 5 words]
User: Something expensive in the west
→ [PREDICTED] System: I found expensive restaurants in the west area available tonight.
User: Make a reservation for 8pm

EXAMPLE 3 - TRANSPORT BOOKING:
User: I need a train
System: [MASKED - 6 words]
User: To London on Friday
→ [PREDICTED] System: I have trains to London available on Friday. What time?
User: Book for 2 people

EXAMPLE 4 - ATTRACTION VISIT:
User: I want to visit the museum
System: [MASKED - 8 words]
User: What are the opening hours?
→ [PREDICTED] System: The museum is open XXXXXXX to XXXXXXX. Entrance is £XXXXXXX.
User: How much are tickets?
"""
}