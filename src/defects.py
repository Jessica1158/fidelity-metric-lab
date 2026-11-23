import numpy as np
import pandas as pd


def inject_template_repetition(
    df,
    severity=0.30,
    seed=42,
    text_col="text",
    label_col="label"
):
    """
    Replace a controlled fraction of messages with repeated
    messages from the same class.

    Returns
    -------
    defective : DataFrame
        Copy of the original dataset with injected repetition.
    audit : DataFrame
        Records which rows changed and their original text.
    """

    if not 0 <= severity <= 1:
        raise ValueError("severity must be between 0 and 1")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError("The text and label columns must exist")

    clean = df.copy().reset_index(drop=True)

    if clean.empty:
        raise ValueError("The dataset is empty")

    if clean[[text_col, label_col]].isna().any().any():
        raise ValueError(
            "Text and label columns cannot contain missing values"
        )

    if not clean[text_col].map(
        lambda x: isinstance(x, str)
    ).all():
        raise ValueError("All messages must be strings")

    defective = clean.copy()
    rng = np.random.default_rng(seed)

    # Group rows by class, e.g. ham and spam.
    groups = [
        (label, np.asarray(indices, dtype=int))
        for label, indices in clean.groupby(
            label_col, sort=False
        ).indices.items()
    ]

    # Allocate replacements proportionally by class.
    # Total = round(severity * dataset_size).
    raw = np.array([
        severity * len(indices)
        for _, indices in groups
    ])

    quotas = np.floor(raw).astype(int)

    remaining = (
        int(round(severity * len(clean)))
        - quotas.sum()
    )

    order = np.argsort(
        -(raw - quotas),
        kind="stable"
    )

    for j in order[:remaining]:
        quotas[j] += 1

    # Audit table.
    audit = pd.DataFrame({
        "row_id": np.arange(len(clean)),
        "label": clean[label_col],
        "original_text": clean[text_col],
        "injected": False,
        "template_source_row": pd.Series(
            pd.NA,
            index=clean.index,
            dtype="Int64"
        )
    })

    for (label, indices), k in zip(groups, quotas):

        if k == 0:
            continue

        texts = clean.loc[indices, text_col].to_numpy()

        # Choose a real message that is not already too frequent.
        # This ensures enough other rows can be replaced.
        counts = pd.Series(texts).value_counts()

        candidates = counts[
            counts <= len(indices) - k
        ]

        if candidates.empty:
            raise ValueError(
                f"Not enough distinct messages in class "
                f"{label!r} to inject severity {severity}."
            )

        template = rng.choice(
            candidates.index.to_numpy()
        )

        # Record which original row supplied the template.
        source_candidates = indices[texts == template]
        source_row = int(rng.choice(source_candidates))

        # Do not select rows already containing the template.
        eligible = indices[texts != template]

        selected = rng.choice(
            eligible,
            size=k,
            replace=False
        )

        # Inject repetition.
        defective.loc[selected, text_col] = template

        # Record changes.
        audit.loc[selected, "injected"] = True

        audit.loc[
            selected,
            "template_source_row"
        ] = source_row

    audit["defective_text"] = defective[text_col]

    return defective, audit

def make_broken_timing_plan(
    campaigns,
    seed=42
):
    """
    Create a fixed Broken Timing injection plan.

    One transition is selected per campaign. The same
    transition can later be reused at every severity level.
    """

    import numpy as np
    import pandas as pd

    required_columns = {
        "campaign_id",
        "message_order",
        "timestamp"
    }

    missing = required_columns - set(campaigns.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rng = np.random.default_rng(seed)

    plan = []

    for campaign_id, group in campaigns.groupby(
        "campaign_id"
    ):

        group = (
            group
            .sort_values("message_order")
            .copy()
        )

        if len(group) < 2:
            raise ValueError(
                f"Campaign {campaign_id} has fewer than 2 messages."
            )

        possible_orders = (
            group["message_order"]
            .iloc[1:]
            .to_numpy()
        )

        target_order = int(
            rng.choice(possible_orders)
        )

        target_idx = group.index[
            group["message_order"] == target_order
        ][0]

        previous_idx = group.index[
            group["message_order"] == target_order - 1
        ][0]

        target_time = pd.to_datetime(
            group.loc[target_idx, "timestamp"]
        )

        previous_time = pd.to_datetime(
            group.loc[previous_idx, "timestamp"]
        )

        original_gap = (
            target_time - previous_time
        ).total_seconds() / 60.0

        plan.append({
            "campaign_id": campaign_id,
            "target_message_order": target_order,
            "original_gap_minutes": original_gap
        })

    return pd.DataFrame(plan)

def inject_broken_timing(
    campaigns,
    injection_plan,
    target_gap_minutes
):
    """
    Inject one Broken Timing violation per campaign.

    The target message and all subsequent timestamps are
    shifted together. This changes only the selected gap;
    later inter-message gaps remain unchanged.
    """

    import pandas as pd

    if target_gap_minutes >= 0:
        raise ValueError(
            "Broken Timing target gap must be negative."
        )

    defective = campaigns.copy()

    defective["timestamp"] = pd.to_datetime(
        defective["timestamp"]
    )

    audit_records = []

    for _, plan_row in injection_plan.iterrows():

        campaign_id = plan_row["campaign_id"]

        target_order = int(
            plan_row["target_message_order"]
        )

        mask_campaign = (
            defective["campaign_id"]
            == campaign_id
        )

        campaign = (
            defective[mask_campaign]
            .sort_values("message_order")
        )

        target_row = campaign[
            campaign["message_order"]
            == target_order
        ].iloc[0]

        previous_row = campaign[
            campaign["message_order"]
            == target_order - 1
        ].iloc[0]

        original_target_time = (
            target_row["timestamp"]
        )

        previous_time = (
            previous_row["timestamp"]
        )

        original_gap = (
            original_target_time
            - previous_time
        ).total_seconds() / 60.0

        # Desired new timestamp.
        new_target_time = (
            previous_time
            + pd.Timedelta(
                minutes=target_gap_minutes
            )
        )

        shift_minutes = (
            new_target_time
            - original_target_time
        ).total_seconds() / 60.0

        # Shift target and all following messages together.
        suffix_mask = (
            mask_campaign
            &
            (
                defective["message_order"]
                >= target_order
            )
        )

        defective.loc[
            suffix_mask,
            "timestamp"
        ] = (
            defective.loc[
                suffix_mask,
                "timestamp"
            ]
            + pd.Timedelta(
                minutes=shift_minutes
            )
        )

        audit_records.append({
            "campaign_id": campaign_id,
            "target_message_order": target_order,
            "original_gap_minutes": original_gap,
            "target_gap_minutes": target_gap_minutes,
            "shift_minutes": shift_minutes,
            "original_target_timestamp":
                original_target_time,
            "new_target_timestamp":
                new_target_time
        })

    audit = pd.DataFrame(
        audit_records
    )

    return defective, audit


def make_entity_inconsistency_plan(
    campaigns,
    entity_catalog,
    seed=42
):
    """
    Create a fixed nested-ready injection plan for
    Entity Inconsistency.

    For each campaign:
    - select one later message containing the canonical name
    - create an inconsistent version by changing only
      the surname
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    required_campaign_columns = {
        "campaign_id",
        "message_id",
        "message_order",
        "text",
        "entity_id"
    }

    missing = (
        required_campaign_columns
        - set(campaigns.columns)
    )

    if missing:
        raise ValueError(
            f"Missing campaign columns: {sorted(missing)}"
        )

    required_entity_columns = {
        "entity_id",
        "canonical_name"
    }

    missing = (
        required_entity_columns
        - set(entity_catalog.columns)
    )

    if missing:
        raise ValueError(
            f"Missing entity catalog columns: {sorted(missing)}"
        )

    entity_lookup = (
        entity_catalog
        .set_index("entity_id")[
            "canonical_name"
        ]
        .to_dict()
    )

    # Pool of available surnames.
    surnames = sorted({
        name.split()[-1]
        for name in entity_catalog[
            "canonical_name"
        ]
    })

    plan_records = []

    for campaign_id, group in campaigns.groupby(
        "campaign_id"
    ):

        group = (
            group
            .sort_values("message_order")
            .copy()
        )

        entity_ids = (
            group["entity_id"]
            .dropna()
            .unique()
        )

        if len(entity_ids) != 1:
            raise ValueError(
                f"{campaign_id} does not have exactly "
                "one canonical entity."
            )

        entity_id = entity_ids[0]

        canonical_name = (
            entity_lookup[entity_id]
        )

        parts = canonical_name.split()

        if len(parts) < 2:
            raise ValueError(
                f"Canonical name must contain "
                f"at least two parts: {canonical_name}"
            )

        first_part = " ".join(parts[:-1])
        original_surname = parts[-1]

        # Later messages containing the full
        # canonical name.
        eligible = group[
            (group["message_order"] > 1)
            &
            (
                group["text"]
                .str.contains(
                    canonical_name,
                    regex=False
                )
            )
        ]

        if eligible.empty:
            raise ValueError(
                f"No later entity mention found "
                f"in campaign {campaign_id}."
            )

        target_row = eligible.iloc[
            int(
                rng.integers(
                    0,
                    len(eligible)
                )
            )
        ]

        replacement_candidates = [
            surname
            for surname in surnames
            if surname != original_surname
        ]

        replacement_surname = (
            rng.choice(
                replacement_candidates
            )
        )

        inconsistent_name = (
            f"{first_part} "
            f"{replacement_surname}"
        )

        plan_records.append({
            "campaign_id":
                campaign_id,

            "entity_id":
                entity_id,

            "target_message_id":
                target_row["message_id"],

            "target_message_order":
                int(
                    target_row[
                        "message_order"
                    ]
                ),

            "canonical_name":
                canonical_name,

            "inconsistent_name":
                inconsistent_name
        })

    plan = pd.DataFrame(
        plan_records
    )

    # One fixed ordering determines nested severity.
    nested_order = rng.permutation(
        len(plan)
    )

    plan["severity_rank"] = 0

    plan.loc[
        nested_order,
        "severity_rank"
    ] = np.arange(
        1,
        len(plan) + 1
    )

    return plan

def inject_entity_inconsistency(
    campaigns,
    injection_plan,
    severity
):
    """
    Inject Entity Inconsistency into a fixed proportion
    of campaigns.

    Severity represents the proportion of campaigns
    containing exactly one inconsistent entity mention.
    """


    if not 0 <= severity <= 1:
        raise ValueError(
            "severity must be between 0 and 1."
        )

    defective = campaigns.copy()

    n_campaigns = (
        injection_plan[
            "campaign_id"
        ].nunique()
    )

    n_affected = int(
        round(
            severity
            * n_campaigns
        )
    )

    selected_plan = (
        injection_plan[
            injection_plan[
                "severity_rank"
            ] <= n_affected
        ]
        .copy()
    )

    audit_records = []

    for _, row in selected_plan.iterrows():

        target_mask = (
            defective["message_id"]
            == row["target_message_id"]
        )

        if target_mask.sum() != 1:
            raise ValueError(
                f"Target message "
                f"{row['target_message_id']} "
                "was not found uniquely."
            )

        original_text = (
            defective.loc[
                target_mask,
                "text"
            ].iloc[0]
        )

        canonical_name = (
            row["canonical_name"]
        )

        inconsistent_name = (
            row["inconsistent_name"]
        )

        if canonical_name not in original_text:
            raise ValueError(
                f"Canonical name not found in "
                f"{row['target_message_id']}."
            )

        defective_text = (
            original_text.replace(
                canonical_name,
                inconsistent_name,
                1
            )
        )

        defective.loc[
            target_mask,
            "text"
        ] = defective_text

        audit_records.append({
            "campaign_id":
                row["campaign_id"],

            "entity_id":
                row["entity_id"],

            "target_message_id":
                row["target_message_id"],

            "target_message_order":
                row["target_message_order"],

            "canonical_name":
                canonical_name,

            "inconsistent_name":
                inconsistent_name,

            "original_text":
                original_text,

            "defective_text":
                defective_text
        })

    audit = pd.DataFrame(
        audit_records
    )

    return defective, audit


import re


def find_missing_cross_channel_candidates(
    campaigns
):
    """
    Find valid candidate injections for the
    Missing Cross-Channel Reference defect.

    A candidate must:

    1. Contain an existing valid cross-channel reference.
    2. Refer to an earlier message.
    3. Explicitly mention the referenced channel in text.
    4. Have an alternative legitimate channel that:
       - differs from the original referenced channel,
       - differs from the current message channel,
       - has no earlier message from the same sender.

    The function only identifies candidates.
    It does not modify the dataset.
    """

    required_columns = {
        "campaign_id",
        "message_id",
        "message_order",
        "channel",
        "sender",
        "text",
        "reference_id"
    }

    missing = (
        required_columns
        - set(campaigns.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    available_channels = sorted(
        campaigns[
            "channel"
        ]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    candidate_records = []

    for campaign_id, group in campaigns.groupby(
        "campaign_id"
    ):

        group = (
            group
            .sort_values(
                "message_order"
            )
            .copy()
        )

        message_lookup = (
            group
            .set_index(
                "message_id"
            )
        )

        for _, target in group.iterrows():

            reference_id = (
                target[
                    "reference_id"
                ]
            )

            if pd.isna(reference_id):
                continue

            if (
                reference_id
                not in message_lookup.index
            ):
                continue

            reference = (
                message_lookup.loc[
                    reference_id
                ]
            )

            target_order = int(
                target[
                    "message_order"
                ]
            )

            reference_order = int(
                reference[
                    "message_order"
                ]
            )

            # Reference must point backward.
            if (
                reference_order
                >= target_order
            ):
                continue

            target_channel = str(
                target["channel"]
            ).lower()

            original_channel = str(
                reference["channel"]
            ).lower()

            # Must originally be cross-channel.
            if (
                original_channel
                == target_channel
            ):
                continue

            text = str(
                target["text"]
            )

            # The original channel must be
            # explicitly mentioned in the text.
            pattern = (
                rf"\b"
                rf"{re.escape(original_channel)}"
                rf"\b"
            )

            if not re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):
                continue

            target_sender = (
                target["sender"]
            )

            previous_messages = group[
                group[
                    "message_order"
                ] < target_order
            ]

            replacement_candidates = []

            for replacement_channel in (
                available_channels
            ):

                if (
                    replacement_channel
                    == original_channel
                ):
                    continue

                if (
                    replacement_channel
                    == target_channel
                ):
                    continue

                # Would a prior message from
                # the same sender actually
                # support this new reference?
                prior_match = (
                    (
                        previous_messages[
                            "channel"
                        ]
                        .astype(str)
                        .str.lower()
                        ==
                        replacement_channel
                    )
                    &
                    (
                        previous_messages[
                            "sender"
                        ]
                        ==
                        target_sender
                    )
                ).any()

                # We specifically want there
                # to be NO valid earlier match.
                if not prior_match:

                    replacement_candidates.append(
                        replacement_channel
                    )

            for replacement_channel in (
                replacement_candidates
            ):

                candidate_records.append({
                    "campaign_id":
                        campaign_id,

                    "target_message_id":
                        target[
                            "message_id"
                        ],

                    "target_message_order":
                        target_order,

                    "reference_id":
                        reference_id,

                    "reference_message_order":
                        reference_order,

                    "target_channel":
                        target_channel,

                    "original_reference_channel":
                        original_channel,

                    "replacement_channel":
                        replacement_channel,

                    "sender":
                        target_sender,

                    "original_text":
                        text
                })

    return pd.DataFrame(
        candidate_records
    )


def make_missing_cross_channel_plan(
    campaigns,
    seed=42
):
    """
    Create a fixed injection plan for
    Missing Cross-Channel Reference.

    One eligible target is selected per campaign.
    A fixed random ranking is then used to create
    nested prevalence levels.
    """

    rng = np.random.default_rng(
        seed
    )

    candidates = (
        find_missing_cross_channel_candidates(
            campaigns
        )
    )

    campaign_ids = (
        campaigns[
            "campaign_id"
        ]
        .unique()
    )

    n_total_campaigns = len(
        campaign_ids
    )

    required_for_max_severity = int(
        round(
            0.60
            * n_total_campaigns
        )
    )

    eligible_campaigns = (
        candidates[
            "campaign_id"
        ].nunique()
    )

    if (
        eligible_campaigns
        <
        required_for_max_severity
    ):
        raise ValueError(
            "Not enough eligible campaigns. "
            f"Found {eligible_campaigns}, "
            f"need at least "
            f"{required_for_max_severity}."
        )

    plan_records = []

    for campaign_id, group in (
        candidates.groupby(
            "campaign_id"
        )
    ):

        group = (
            group
            .reset_index(
                drop=True
            )
        )

        selected_index = int(
            rng.integers(
                0,
                len(group)
            )
        )

        selected = (
            group.iloc[
                selected_index
            ]
            .to_dict()
        )

        plan_records.append(
            selected
        )

    plan = pd.DataFrame(
        plan_records
    )

    # Fixed ordering for nested severity.
    permutation = (
        rng.permutation(
            len(plan)
        )
    )

    plan["severity_rank"] = 0

    plan.loc[
        permutation,
        "severity_rank"
    ] = np.arange(
        1,
        len(plan) + 1
    )

    return plan

def inject_missing_cross_channel_reference(
    campaigns,
    injection_plan,
    severity
):
    """
    Inject Missing Cross-Channel Reference.

    Severity is the proportion of ALL campaigns
    that receive exactly one corrupted textual
    cross-channel reference.

    Only the target message text is modified.
    """

    if not 0 <= severity <= 1:
        raise ValueError(
            "severity must be between 0 and 1."
        )

    defective = (
        campaigns.copy()
    )

    n_total_campaigns = (
        campaigns[
            "campaign_id"
        ].nunique()
    )

    n_affected = int(
        round(
            severity
            * n_total_campaigns
        )
    )

    selected_plan = (
        injection_plan[
            injection_plan[
                "severity_rank"
            ]
            <= n_affected
        ]
        .copy()
    )

    audit_records = []

    for _, row in (
        selected_plan.iterrows()
    ):

        target_mask = (
            defective[
                "message_id"
            ]
            ==
            row[
                "target_message_id"
            ]
        )

        if target_mask.sum() != 1:
            raise ValueError(
                "Target message not "
                "found uniquely."
            )

        original_text = (
            defective.loc[
                target_mask,
                "text"
            ].iloc[0]
        )

        original_channel = (
            row[
                "original_reference_channel"
            ]
        )

        replacement_channel = (
            row[
                "replacement_channel"
            ]
        )

        pattern = (
            rf"\b"
            rf"{re.escape(original_channel)}"
            rf"\b"
        )

        defective_text, n_replacements = (
            re.subn(
                pattern,
                replacement_channel,
                original_text,
                count=1,
                flags=re.IGNORECASE
            )
        )

        if n_replacements != 1:
            raise ValueError(
                "Expected exactly one "
                "channel replacement."
            )

        defective.loc[
            target_mask,
            "text"
        ] = defective_text

        audit_records.append({
            "campaign_id":
                row[
                    "campaign_id"
                ],

            "target_message_id":
                row[
                    "target_message_id"
                ],

            "target_message_order":
                row[
                    "target_message_order"
                ],

            "reference_id":
                row[
                    "reference_id"
                ],

            "original_reference_channel":
                original_channel,

            "replacement_channel":
                replacement_channel,

            "original_text":
                original_text,

            "defective_text":
                defective_text
        })

    audit = pd.DataFrame(
        audit_records
    )

    return defective, audit

import re
import pandas as pd


OVER_POLISHED_REPLACEMENTS = [
    # SMS abbreviations
    (r"\bpls\b", "please"),
    (r"\bplz\b", "please"),
    (r"\bthx\b", "thank you"),
    (r"\bthanx\b", "thank you"),
    (r"\btks\b", "thank you"),
    (r"\bmsg\b", "message"),
    (r"\btxt\b", "text"),
    (r"\btmrw\b", "tomorrow"),
    (r"\btmr\b", "tomorrow"),
    (r"\bnite\b", "night"),
    (r"\basap\b", "as soon as possible"),
    (r"\bb4\b", "before"),
    (r"\bl8r\b", "later"),
    (r"\bgr8\b", "great"),
    (r"\bu\b", "you"),
    (r"\bur\b", "your"),
    (r"\br\b", "are"),

    # Informal forms
    (r"\bwanna\b", "would like to"),
    (r"\bgonna\b", "going to"),
    (r"\bgotta\b", "need to"),

    # Contractions / informal contractions
    (r"\bcan't\b", "cannot"),
    (r"\bcant\b", "cannot"),
    (r"\bdon't\b", "do not"),
    (r"\bdont\b", "do not"),
    (r"\bwon't\b", "will not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),

    (r"\bi'm\b", "I am"),
    (r"\bim\b", "I am"),
    (r"\bi've\b", "I have"),
    (r"\bive\b", "I have"),
    (r"\bi'll\b", "I will"),
    (r"\bill\b", "I will"),

    (r"\byou're\b", "you are"),
    (r"\byoure\b", "you are"),
    (r"\byou've\b", "you have"),
    (r"\byou'll\b", "you will"),

    (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"),

    # Mild formalization
    (r"\bhey\b", "Hello"),
    (r"\bhiya\b", "Hello"),
    (r"\bthanks\b", "Thank you"),
]


def _protect_sms_spans(text):
    """
    Protect URLs, e-mail addresses, phone-like strings,
    currency amounts and numeric expressions so that
    stylistic rewriting cannot modify them.
    """

    patterns = [
        r"https?://\S+",
        r"www\.\S+",
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        r"\+?\d[\d\s\-\(\)]{6,}\d",
        r"[£$€]\s?\d+(?:[.,]\d+)*",
        r"\b\d+(?:[.,]\d+)*\b",
    ]

    protected = {}
    counter = 0

    def replace_match(match):
        nonlocal counter

        key = f"__PROTECTED_{counter}__"
        protected[key] = match.group(0)
        counter += 1

        return key

    protected_text = str(text)

    for pattern in patterns:
        protected_text = re.sub(
            pattern,
            replace_match,
            protected_text
        )

    return protected_text, protected


def _restore_sms_spans(text, protected):
    """
    Restore protected spans exactly.
    """

    for key, value in protected.items():
        text = text.replace(
            key,
            value
        )

    return text


def over_polish_sms_text(text):
    """
    Apply a deterministic style-only transformation.

    The transformation:
    - expands common SMS shorthand,
    - expands contractions,
    - applies mild formal lexical substitutions,
    - normalizes capitalization and punctuation,
    - preserves protected factual spans.

    Returns
    -------
    polished_text : str
    n_style_changes : int
    """

    original_text = str(text)

    working_text, protected = (
        _protect_sms_spans(
            original_text
        )
    )

    n_changes = 0

    # -----------------------------------------
    # Lexical / stylistic replacements
    # -----------------------------------------

    for pattern, replacement in (
        OVER_POLISHED_REPLACEMENTS
    ):

        working_text, count = re.subn(
            pattern,
            replacement,
            working_text,
            flags=re.IGNORECASE
        )

        n_changes += count

    # -----------------------------------------
    # Normalize whitespace
    # -----------------------------------------

    normalized = re.sub(
        r"\s+",
        " ",
        working_text
    ).strip()

    if normalized != working_text.strip():
        n_changes += 1

    working_text = normalized

    # -----------------------------------------
    # Reduce excessive punctuation
    # -----------------------------------------

    normalized = re.sub(
        r"!{2,}",
        "!",
        working_text
    )

    normalized = re.sub(
        r"\?{2,}",
        "?",
        normalized
    )

    if normalized != working_text:
        n_changes += 1

    working_text = normalized

    # -----------------------------------------
    # Capitalize beginning of message/sentences
    # -----------------------------------------

    def capitalize_match(match):

        return (
            match.group(1)
            +
            match.group(2).upper()
        )

    capitalized = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        capitalize_match,
        working_text
    )

    if capitalized != working_text:
        n_changes += 1

    working_text = capitalized

    # -----------------------------------------
    # Add final punctuation when appropriate
    # -----------------------------------------

    if (
        working_text
        and working_text[-1]
        not in ".!?"
    ):
        working_text += "."
        n_changes += 1

    # -----------------------------------------
    # Restore protected information
    # -----------------------------------------

    polished_text = (
        _restore_sms_spans(
            working_text,
            protected
        )
    )

    return polished_text, n_changes

def find_over_polished_candidates(
    sms_df
):
    """
    Return messages receiving at least one explicit
    stylistic transformation under the frozen
    Over-Polished Language transformation.
    """

    required_columns = {
        "message_id",
        "label",
        "text"
    }

    missing = (
        required_columns
        - set(sms_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    records = []

    for _, row in sms_df.iterrows():

        polished_text, n_changes = (
            over_polish_sms_text(
                row["text"]
            )
        )

        # Require at least one actual
        # tracked stylistic intervention.
        if (
            polished_text != row["text"]
            and n_changes > 0
        ):

            records.append({
                "message_id":
                    row["message_id"],

                "label":
                    row["label"],

                "original_text":
                    row["text"],

                "polished_text":
                    polished_text,

                "n_style_changes":
                    n_changes
            })

    return pd.DataFrame(
        records
    )

def make_over_polished_plan(
    sms_eval,
    seed=42
):
    """
    Create a fixed, class-stratified, nested injection
    plan for Over-Polished Language.

    Severity will later be controlled only through
    prevalence.

    The plan preserves approximately the ham/spam
    composition of the evaluation set.
    """

    rng = np.random.default_rng(
        seed
    )

    candidates = (
        find_over_polished_candidates(
            sms_eval
        )
    )

    # Required counts for maximum severity:
    # 60% of 300 = 180 messages
    # Preserve evaluation class proportions:
    # 156 ham + 24 spam
    required_max = {
        "ham": 156,
        "spam": 24
    }

    for label, required in (
        required_max.items()
    ):

        available = (
            candidates[
                candidates["label"]
                == label
            ].shape[0]
        )

        if available < required:

            raise ValueError(
                f"Not enough {label} candidates: "
                f"{available} available, "
                f"{required} required."
            )

    selected_records = []

    for label in [
        "ham",
        "spam"
    ]:

        group = (
            candidates[
                candidates[
                    "label"
                ] == label
            ]
            .copy()
            .reset_index(drop=True)
        )

        permutation = (
            rng.permutation(
                len(group)
            )
        )

        group = (
            group.iloc[
                permutation
            ]
            .reset_index(drop=True)
        )

        group[
            "class_rank"
        ] = np.arange(
            1,
            len(group) + 1
        )

        selected_records.append(
            group
        )

    plan = pd.concat(
        selected_records,
        ignore_index=True
    )

    return plan

def inject_over_polished_language(
    sms_eval,
    injection_plan,
    severity
):
    """
    Inject Over-Polished Language using fixed,
    class-stratified nested subsets.

    Only the text field is modified.

    Supported severity levels:
        0.10
        0.30
        0.60
    """

    severity_counts = {
        0.10: {
            "ham": 26,
            "spam": 4
        },
        0.30: {
            "ham": 78,
            "spam": 12
        },
        0.60: {
            "ham": 156,
            "spam": 24
        }
    }

    if severity not in severity_counts:

        raise ValueError(
            "severity must be one of "
            "0.10, 0.30, or 0.60."
        )

    defective = (
        sms_eval
        .copy()
        .reset_index(drop=True)
    )

    selected_parts = []

    for label, n_required in (
        severity_counts[
            severity
        ].items()
    ):

        selected = (
            injection_plan[
                (
                    injection_plan[
                        "label"
                    ] == label
                )
                &
                (
                    injection_plan[
                        "class_rank"
                    ]
                    <= n_required
                )
            ]
            .copy()
        )

        selected_parts.append(
            selected
        )

    selected_plan = pd.concat(
        selected_parts,
        ignore_index=True
    )

    audit_records = []

    for _, row in (
        selected_plan.iterrows()
    ):

        mask = (
            defective[
                "message_id"
            ]
            ==
            row[
                "message_id"
            ]
        )

        if mask.sum() != 1:

            raise ValueError(
                "Target message not "
                "found uniquely."
            )

        original_text = (
            defective.loc[
                mask,
                "text"
            ].iloc[0]
        )

        defective.loc[
            mask,
            "text"
        ] = row[
            "polished_text"
        ]

        audit_records.append({
            "message_id":
                row[
                    "message_id"
                ],

            "label":
                row[
                    "label"
                ],

            "n_style_changes":
                row[
                    "n_style_changes"
                ],

            "original_text":
                original_text,

            "polished_text":
                row[
                    "polished_text"
                ]
        })

    audit = pd.DataFrame(
        audit_records
    )

    return defective, audit

def make_template_repetition_plan(eval_df,reference_df,severities=(0.10,0.30,0.60),seed=42,id_col="message_id",label_col="label",text_col="text"):
    import numpy as np,pandas as pd
    rng=np.random.default_rng(seed)
    templates={}
    orders={}
    for label in sorted(eval_df[label_col].unique()):
        ref=reference_df.loc[reference_df[label_col]==label].reset_index(drop=True)
        source=ref.iloc[rng.integers(len(ref))]
        candidates=eval_df.loc[(eval_df[label_col]==label)&(eval_df[text_col]!=source[text_col]),id_col].to_numpy().copy()
        rng.shuffle(candidates)
        templates[label]={"source_id":source[id_col],"template_text":source[text_col]}
        orders[label]=candidates
    records=[]
    for severity in severities:
        for label in sorted(orders):
            n=int(round((eval_df[label_col]==label).sum()*severity))
            if n>len(orders[label]): raise ValueError(f"Not enough eligible {label} messages.")
            for target_id in orders[label][:n]:
                records.append({"severity":float(severity),"message_id":target_id,"label":label,"template_source_id":templates[label]["source_id"],"template_text":templates[label]["template_text"]})
    return pd.DataFrame(records)

def apply_template_repetition_plan(eval_df,plan,severity,id_col="message_id",text_col="text"):
    import numpy as np
    out=eval_df.copy()
    p=plan.loc[np.isclose(plan["severity"],severity)]
    replacements=dict(zip(p[id_col],p["template_text"]))
    mask=out[id_col].isin(replacements)
    out.loc[mask,text_col]=out.loc[mask,id_col].map(replacements)
    return out

def make_broken_timing_plan(eval_df,severities=(0.10,0.30,0.60),target_gap_minutes=-5.0,seed=42):
    import numpy as np,pandas as pd
    rng=np.random.default_rng(seed)
    campaign_ids=np.array(sorted(eval_df["campaign_id"].unique()))
    rng.shuffle(campaign_ids)
    targets={}
    for cid in campaign_ids:
        g=eval_df.loc[eval_df["campaign_id"]==cid].sort_values("message_order")
        orders=g["message_order"].to_numpy()
        target=int(rng.choice(orders[1:]))
        current=pd.to_datetime(g.loc[g["message_order"]==target,"timestamp"].iloc[0])
        previous=pd.to_datetime(g.loc[g["message_order"]==target-1,"timestamp"].iloc[0])
        targets[cid]={"target_message_order":target,"original_gap_minutes":float((current-previous).total_seconds()/60)}
    records=[]
    for severity in severities:
        n=int(round(len(campaign_ids)*severity))
        for cid in campaign_ids[:n]:
            records.append({"severity":float(severity),"campaign_id":cid,**targets[cid],"target_gap_minutes":float(target_gap_minutes)})
    return pd.DataFrame(records)

def apply_broken_timing_plan(eval_df,plan,severity):
    import numpy as np,pandas as pd
    out=eval_df.copy()
    out["timestamp"]=pd.to_datetime(out["timestamp"])
    p=plan.loc[np.isclose(plan["severity"],severity)]
    for row in p.itertuples(index=False):
        mask=out["campaign_id"]==row.campaign_id
        g=out.loc[mask].sort_values("message_order")
        prev_ts=g.loc[g["message_order"]==row.target_message_order-1,"timestamp"].iloc[0]
        cur_ts=g.loc[g["message_order"]==row.target_message_order,"timestamp"].iloc[0]
        shift=(cur_ts-prev_ts)-pd.Timedelta(minutes=row.target_gap_minutes)
        later=mask&(out["message_order"]>=row.target_message_order)
        out.loc[later,"timestamp"]=out.loc[later,"timestamp"]-shift
    return out