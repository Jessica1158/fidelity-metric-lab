import numpy as np
import pandas as pd
from itertools import product


CHANNELS = np.array([
    "email",
    "sms",
    "whatsapp"
])


FIRST_NAMES = [
    "Alex", "Daniel", "Emma", "Sofia", "Lucas",
    "Mia", "David", "Olivia", "Carlos", "Laura",
    "Martin", "Elena", "Samuel", "Julia", "Victor",
    "Andrea", "Diego", "Natalia", "Thomas", "Isabel",
    "Gabriel", "Clara", "Marco", "Valeria", "Adrian",
    "Paula", "Leo", "Camila", "Eric", "Diana",
    "Oscar", "Sara"
]


LAST_NAMES = [
    "Morgan", "Parker", "Turner", "Collins", "Reed",
    "Bennett", "Foster", "Cooper", "Bailey", "Howard",
    "Ward", "Gray", "Morris", "Price", "Kelly",
    "Rivera", "Torres", "Flores", "Cruz", "Vega",
    "Navarro", "Mendoza", "Castro", "Ramos", "Ortega",
    "Silva", "Lopez", "Ruiz", "Santos", "Herrera",
    "Medina", "Campos"
]


def generate_clean_campaigns(
    n_campaigns=1000,
    seed=42
):
    """
    Generate clean structured communication campaigns.

    Each campaign contains:
    - 5 to 9 messages
    - at least two channels
    - strictly increasing timestamps
    - one consistent main entity
    - at least one valid cross-channel reference

    Returns
    -------
    campaigns_df : DataFrame
        One row per message.

    entity_catalog_df : DataFrame
        Ground-truth mapping between entity IDs and
        canonical entity names.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------
    # Create possible synthetic names
    # --------------------------------------------

    names = [
        f"{first} {last}"
        for first, last
        in product(FIRST_NAMES, LAST_NAMES)
    ]

    if n_campaigns > len(names):
        raise ValueError(
            "Not enough unique synthetic names "
            "for the requested number of campaigns."
        )

    rng.shuffle(names)

    records = []
    entity_records = []

    # --------------------------------------------
    # Message template pools
    # --------------------------------------------

    opening_templates = [
        "Hello {name}, I am sending the information we discussed.",
        "Hi {name}, here is the summary for your review.",
        "Hello {name}, I am sharing the initial details we discussed."
    ]

    acknowledgement_templates = [
        "Thanks, I received the information.",
        "Thank you. I will review it.",
        "I received it, thanks. I will take a look."
    ]

    question_templates = [
        "I reviewed the information and I have a question about the details.",
        "I have reviewed it. Could you clarify one of the points?",
        "I looked through the information and have a quick question."
    ]

    advisor_response_templates = [
        "Of course, {name}. I can clarify that point for you.",
        "Certainly, {name}. I will explain that in more detail.",
        "No problem, {name}. Let me clarify the information."
    ]

    client_followups = [
        "Thanks, that answers my question.",
        "That makes sense. Thank you.",
        "Understood. I appreciate the clarification.",
        "Thanks. I will review the remaining information."
    ]

    advisor_followups = [
        "Let me know if you need any additional information, {name}.",
        "I am available if you have any other questions, {name}.",
        "Feel free to contact me if anything else comes up, {name}.",
        "I can provide more information if needed, {name}."
    ]

    # --------------------------------------------
    # Generate campaigns
    # --------------------------------------------

    for i in range(n_campaigns):

        campaign_id = f"C{i + 1:04d}"
        entity_id = f"E{i + 1:04d}"

        client_id = f"client_{i + 1:04d}"

        advisor_number = int(
            rng.integers(1, 21)
        )

        advisor_id = (
            f"advisor_{advisor_number:02d}"
        )

        canonical_name = names[i]

        entity_records.append({
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "client_id": client_id
        })

        # ----------------------------------------
        # Number of messages
        # ----------------------------------------

        n_messages = int(
            rng.integers(5, 10)
        )

        message_ids = [
            f"{campaign_id}_M{j + 1:02d}"
            for j in range(n_messages)
        ]

        # ----------------------------------------
        # Choose channels
        # ----------------------------------------

        first_channel = rng.choice(CHANNELS)

        other_channels = CHANNELS[
            CHANNELS != first_channel
        ]

        # Message 3 is guaranteed to use
        # a different channel from message 1.
        reference_channel = rng.choice(
            other_channels
        )

        channel_sequence = [
            first_channel,
            rng.choice(CHANNELS),
            reference_channel,
            rng.choice(CHANNELS),
            rng.choice(CHANNELS)
        ]

        while len(channel_sequence) < n_messages:
            channel_sequence.append(
                rng.choice(CHANNELS)
            )

        # ----------------------------------------
        # Construct message texts
        # ----------------------------------------

        texts = []

        texts.append(
            rng.choice(
                opening_templates
            ).format(
                name=canonical_name
            )
        )

        texts.append(
            rng.choice(
                acknowledgement_templates
            )
        )

        # Explicit cross-channel reference.
        texts.append(
            (
                f"Following up here on the "
                f"{first_channel} message I sent earlier, "
                f"{canonical_name}. Let me know if anything "
                f"is unclear."
            )
        )

        texts.append(
            rng.choice(
                question_templates
            )
        )

        texts.append(
            rng.choice(
                advisor_response_templates
            ).format(
                name=canonical_name
            )
        )

        # ----------------------------------------
        # Additional messages
        # ----------------------------------------

        for j in range(5, n_messages):

            if j % 2 == 1:

                text = rng.choice(
                    client_followups
                )

            else:

                text = rng.choice(
                    advisor_followups
                ).format(
                    name=canonical_name
                )

            texts.append(text)

        # ----------------------------------------
        # Sender / recipient sequence
        # ----------------------------------------

        senders = []
        recipients = []

        for j in range(n_messages):

            if j % 2 == 0:

                senders.append(advisor_id)
                recipients.append(client_id)

            else:

                senders.append(client_id)
                recipients.append(advisor_id)

        # ----------------------------------------
        # Generate ordered timestamps
        # ----------------------------------------

        start_day = int(
            rng.integers(0, 180)
        )

        start_hour = int(
            rng.integers(8, 17)
        )

        start_minute = int(
            rng.integers(0, 60)
        )

        current_time = (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=start_day)
            + pd.Timedelta(hours=start_hour)
            + pd.Timedelta(minutes=start_minute)
        )

        timestamps = [current_time]

        for _ in range(1, n_messages):

            gap_minutes = int(
                rng.integers(5, 121)
            )

            current_time = (
                current_time
                + pd.Timedelta(
                    minutes=gap_minutes
                )
            )

            timestamps.append(
                current_time
            )

        # ----------------------------------------
        # Add rows
        # ----------------------------------------

        for j in range(n_messages):

            reference_id = None

            # Message 3 explicitly references
            # message 1 across channels.
            if j == 2:
                reference_id = message_ids[0]

            parent_message_id = None

            if j > 0:
                parent_message_id = (
                    message_ids[j - 1]
                )

            records.append({
                "campaign_id": campaign_id,
                "message_id": message_ids[j],
                "message_order": j + 1,
                "timestamp": timestamps[j],
                "channel": channel_sequence[j],
                "sender": senders[j],
                "recipient": recipients[j],
                "text": texts[j],
                "entity_id": entity_id,
                "reference_id": reference_id,
                "parent_message_id": parent_message_id
            })

    campaigns_df = pd.DataFrame(
        records
    )

    entity_catalog_df = pd.DataFrame(
        entity_records
    )

    return campaigns_df, entity_catalog_df