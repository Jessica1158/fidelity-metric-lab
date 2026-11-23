import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def make_tfidf_vectorizer():
    """
    TF-IDF configuration for the first template-repetition dry run.
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True,
        stop_words=None,
        norm="l2"
    )

def generic_structural_summary(campaign):
    """
    Convert one communication campaign into a fixed
    13-feature generic structural representation.

    Parameters
    ----------
    campaign : pandas.DataFrame
        All messages belonging to one campaign.

    Returns
    -------
    pandas.Series
        The 13 generic structural features.
    """

    required_columns = {
        "campaign_id",
        "message_order",
        "timestamp",
        "channel",
        "text"
    }

    missing = required_columns - set(campaign.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if campaign.empty:
        raise ValueError("Campaign cannot be empty.")

    if campaign["campaign_id"].nunique() != 1:
        raise ValueError(
            "The input must contain exactly one campaign."
        )

    if len(campaign) < 2:
        raise ValueError(
            "A campaign must contain at least two messages."
        )

    # --------------------------------------------------
    # Preserve the logical event order.
    # DO NOT sort by timestamp.
    # --------------------------------------------------

    campaign = (
        campaign
        .sort_values("message_order")
        .copy()
    )

    if campaign["message_order"].duplicated().any():
        raise ValueError(
            "message_order must be unique within a campaign."
        )

    # --------------------------------------------------
    # Timestamp preparation
    # --------------------------------------------------

    timestamps = pd.to_datetime(
        campaign["timestamp"],
        errors="raise"
    )

    # Time gaps are calculated according to message_order.
    gaps_minutes = (
        timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    # Duration describes the overall temporal span.
    campaign_duration = (
        (
            timestamps.max()
            - timestamps.min()
        ).total_seconds()
        / 60.0
    )

    # --------------------------------------------------
    # Channel features
    # --------------------------------------------------

    channels = campaign["channel"].astype(str)

    n_channels = channels.nunique()

    transitions = (
        channels.to_numpy()[1:]
        != channels.to_numpy()[:-1]
    )

    channel_switch_rate = transitions.mean()

    channel_counts = (
        channels
        .value_counts(normalize=True)
        .to_numpy()
    )

    dominant_channel_share = (
        channel_counts.max()
    )

    # Normalized Shannon entropy.
    if n_channels == 1:
        channel_entropy = 0.0

    else:
        entropy = -np.sum(
            channel_counts
            * np.log(channel_counts)
        )

        channel_entropy = (
            entropy
            / np.log(n_channels)
        )

    # --------------------------------------------------
    # Text-size features
    # --------------------------------------------------

    texts = campaign["text"].astype(str)

    message_lengths = (
        texts
        .str.len()
        .to_numpy()
    )

    token_counts = (
        texts
        .apply(lambda text: len(text.split()))
        .to_numpy()
    )

    # --------------------------------------------------
    # Return the fixed 13-feature representation
    # --------------------------------------------------

    return pd.Series({
        "n_messages":
            len(campaign),

        "campaign_duration":
            campaign_duration,

        "mean_time_gap":
            gaps_minutes.mean(),

        "median_time_gap":
            gaps_minutes.median(),

        "std_time_gap":
            gaps_minutes.std(ddof=0),

        "n_channels":
            n_channels,

        "channel_switch_rate":
            channel_switch_rate,

        "channel_entropy":
            channel_entropy,

        "dominant_channel_share":
            dominant_channel_share,

        "mean_message_length":
            message_lengths.mean(),

        "std_message_length":
            message_lengths.std(ddof=0),

        "mean_token_count":
            token_counts.mean(),

        "std_token_count":
            token_counts.std(ddof=0)
    })

def _participant_role(participant):
    """
    Convert synthetic participant identifiers into
    generic communication roles.
    """

    participant = str(participant)

    if participant.startswith("advisor_"):
        return "ADVISOR"

    if participant.startswith("client_"):
        return "CLIENT"

    return "PARTICIPANT"


def serialize_campaign(campaign):
    """
    Convert one campaign into one ordered text string.

    The serialization contains only general observable
    communication information:
        - message order
        - channel
        - sender/recipient role
        - message text

    Ground-truth and defect-specific metadata are excluded.
    """

    required_columns = {
        "campaign_id",
        "message_order",
        "channel",
        "sender",
        "recipient",
        "text"
    }

    missing = required_columns - set(campaign.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if campaign.empty:
        raise ValueError(
            "Campaign cannot be empty."
        )

    if campaign["campaign_id"].nunique() != 1:
        raise ValueError(
            "Input must contain exactly one campaign."
        )

    campaign = (
        campaign
        .sort_values("message_order")
        .copy()
    )

    serialized_messages = []

    for _, row in campaign.iterrows():

        sender_role = _participant_role(
            row["sender"]
        )

        recipient_role = _participant_role(
            row["recipient"]
        )

        channel = str(
            row["channel"]
        ).upper()

        message_order = int(
            row["message_order"]
        )

        text = str(
            row["text"]
        ).strip()

        serialized_message = (
            f"[MESSAGE {message_order}] "
            f"[CHANNEL={channel}] "
            f"[{sender_role}->{recipient_role}] "
            f"{text}"
        )

        serialized_messages.append(
            serialized_message
        )

    return "\n".join(
        serialized_messages
    )


def serialize_campaign_dataset(campaigns):
    """
    Serialize every campaign in a dataset.

    Returns one row per campaign.
    """

    records = []

    for campaign_id, campaign in campaigns.groupby(
        "campaign_id",
        sort=True
    ):

        serialized_text = serialize_campaign(
            campaign
        )

        records.append({
            "campaign_id": campaign_id,
            "serialized_text": serialized_text
        })

    return pd.DataFrame(records)

import numpy as np
import pandas as pd

from time import perf_counter


def split_campaign_into_chunks(
    serialized_text,
    model
):
    """
    Split one serialized campaign into consecutive chunks
    that fit within the model token limit.

    Message boundaries and message order are preserved.
    """

    tokenizer = model.tokenizer
    max_length = model.max_seq_length

    messages = [
        line.strip()
        for line in serialized_text.split("\n")
        if line.strip()
    ]

    chunks = []
    current_messages = []

    for message in messages:

        candidate = "\n".join(
            current_messages + [message]
        )

        token_count = len(
            tokenizer(
                candidate,
                add_special_tokens=True,
                truncation=False
            )["input_ids"]
        )

        if token_count <= max_length:

            current_messages.append(
                message
            )

        else:

            if current_messages:

                chunks.append(
                    "\n".join(
                        current_messages
                    )
                )

                current_messages = [
                    message
                ]

                single_length = len(
                    tokenizer(
                        message,
                        add_special_tokens=True,
                        truncation=False
                    )["input_ids"]
                )

                if single_length > max_length:

                    raise ValueError(
                        "A single message exceeds "
                        "the model token limit."
                    )

            else:

                raise ValueError(
                    "A single message exceeds "
                    "the model token limit."
                )

    if current_messages:

        chunks.append(
            "\n".join(
                current_messages
            )
        )

    return chunks


def encode_serialized_campaigns(
    serialized_df,
    model,
    batch_size=64
):
    """
    Produce one semantic embedding per campaign.

    Long campaigns are split at message boundaries.
    Chunk embeddings are combined using a token-weighted
    mean and the final campaign vector is L2-normalized.

    Parameters
    ----------
    serialized_df : pd.DataFrame
        Must contain a column named 'serialized_text'.

    model
        A SentenceTransformer model.

    batch_size : int
        Encoding batch size.

    Returns
    -------
    campaign_embeddings : np.ndarray
        Matrix with one embedding per campaign.

    runtime : float
        Encoding runtime in seconds.
    """

    required_columns = {
        "serialized_text"
    }

    missing = (
        required_columns
        - set(serialized_df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    all_chunks = []
    chunk_campaign_indices = []
    chunk_weights = []

    # -----------------------------------------
    # Create chunks
    # -----------------------------------------

    for campaign_index, text in enumerate(
        serialized_df[
            "serialized_text"
        ]
    ):

        chunks = split_campaign_into_chunks(
            text,
            model
        )

        for chunk in chunks:

            weight = len(
                model.tokenizer(
                    chunk,
                    add_special_tokens=False,
                    truncation=False
                )["input_ids"]
            )

            all_chunks.append(
                chunk
            )

            chunk_campaign_indices.append(
                campaign_index
            )

            chunk_weights.append(
                weight
            )

    print(
        "Campaigns:",
        len(serialized_df)
    )

    print(
        "Chunks to encode:",
        len(all_chunks)
    )

    # -----------------------------------------
    # Encode chunks
    # -----------------------------------------

    start = perf_counter()

    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # -----------------------------------------
    # Aggregate chunks into campaigns
    # -----------------------------------------

    n_campaigns = len(
        serialized_df
    )

    embedding_dim = (
        chunk_embeddings.shape[1]
    )

    campaign_embeddings = np.zeros(
        (
            n_campaigns,
            embedding_dim
        ),
        dtype=float
    )

    total_weights = np.zeros(
        n_campaigns,
        dtype=float
    )

    for (
        embedding,
        campaign_index,
        weight
    ) in zip(
        chunk_embeddings,
        chunk_campaign_indices,
        chunk_weights
    ):

        campaign_embeddings[
            campaign_index
        ] += (
            embedding * weight
        )

        total_weights[
            campaign_index
        ] += weight

    campaign_embeddings /= (
        total_weights[:, None]
    )

    # -----------------------------------------
    # Final L2 normalization
    # -----------------------------------------

    norms = np.linalg.norm(
        campaign_embeddings,
        axis=1,
        keepdims=True
    )

    campaign_embeddings /= norms

    runtime = (
        perf_counter()
        - start
    )

    return (
        campaign_embeddings,
        runtime
    )

def encode_text_messages(
    message_df,
    model,
    text_column="text",
    batch_size=64,
    normalize_embeddings=True
):
    """
    Encode individual text messages with a frozen
    SentenceTransformer model.

    One input message produces one embedding vector.

    The function checks sequence lengths before
    encoding to prevent silent truncation.

    Returns
    -------
    embeddings : np.ndarray
        One embedding per message.

    runtime : float
        Encoding runtime in seconds.
    """

    import numpy as np

    from time import perf_counter


    if text_column not in message_df.columns:

        raise ValueError(
            f"Missing text column: "
            f"{text_column}"
        )


    texts = (
        message_df[
            text_column
        ]
        .astype(str)
        .tolist()
    )


    # -----------------------------------------
    # Prevent silent truncation
    # -----------------------------------------

    for i, text in enumerate(
        texts
    ):

        n_tokens = len(
            model.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )[
                "input_ids"
            ]
        )

        if (
            n_tokens
            >
            model.max_seq_length
        ):

            raise ValueError(
                f"Message {i} contains "
                f"{n_tokens} tokens, exceeding "
                f"the model limit of "
                f"{model.max_seq_length}."
            )


    # -----------------------------------------
    # Encode
    # -----------------------------------------

    start = perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=(
            normalize_embeddings
        )
    )

    runtime = (
        perf_counter()
        - start
    )


    return embeddings, runtime

def paired_embedding_distances(
    clean_embeddings,
    defective_embeddings,
    tolerance=1e-8
):
    """
    Compute paired representation-level diagnostics
    between clean and defective embeddings.

    Parameters
    ----------
    clean_embeddings : np.ndarray
        Shape (n_samples, n_features).

    defective_embeddings : np.ndarray
        Shape (n_samples, n_features).

    tolerance : float
        Threshold used to identify zero-displacement pairs.

    Returns
    -------
    dict
        Per-pair Euclidean distances, cosine similarities,
        and zero-displacement indicators.
    """

    import numpy as np


    clean_embeddings = np.asarray(
        clean_embeddings
    )

    defective_embeddings = np.asarray(
        defective_embeddings
    )


    if (
        clean_embeddings.shape
        != defective_embeddings.shape
    ):
        raise ValueError(
            "Clean and defective embeddings "
            "must have the same shape."
        )


    # -----------------------------------------
    # Euclidean displacement
    # -----------------------------------------

    euclidean_distance = np.linalg.norm(
        defective_embeddings
        -
        clean_embeddings,
        axis=1
    )


    # -----------------------------------------
    # Cosine similarity
    # -----------------------------------------

    clean_norms = np.linalg.norm(
        clean_embeddings,
        axis=1
    )

    defective_norms = np.linalg.norm(
        defective_embeddings,
        axis=1
    )


    cosine_similarity = np.sum(
        clean_embeddings
        *
        defective_embeddings,
        axis=1
    ) / (
        clean_norms
        *
        defective_norms
    )


    # -----------------------------------------
    # Zero-displacement cases
    # -----------------------------------------

    zero_displacement = (
        euclidean_distance
        <= tolerance
    )


    return {
        "euclidean_distance":
            euclidean_distance,

        "cosine_similarity":
            cosine_similarity,

        "zero_displacement":
            zero_displacement
    }
def paired_embedding_distance_table(
    unit_ids,
    clean_embeddings,
    defective_embeddings,
    affected_ids=None,
    tolerance=1e-8
):
    """
    Build a pair-level representation diagnostic table
    for aligned clean and defective embeddings.

    Parameters
    ----------
    unit_ids : array-like
        IDs corresponding to embedding rows.

    clean_embeddings : np.ndarray
        Clean embedding matrix.

    defective_embeddings : np.ndarray
        Defective embedding matrix.

    affected_ids : set, optional
        Ground-truth IDs affected by the defect.

    tolerance : float
        Threshold for zero displacement.

    Returns
    -------
    pd.DataFrame
        One row per clean/defective pair.
    """

    import numpy as np
    import pandas as pd


    unit_ids = np.asarray(
        unit_ids
    )

    clean_embeddings = np.asarray(
        clean_embeddings
    )

    defective_embeddings = np.asarray(
        defective_embeddings
    )


    if (
        clean_embeddings.shape
        != defective_embeddings.shape
    ):

        raise ValueError(
            "Clean and defective embedding "
            "matrices must have identical shape."
        )


    if (
        clean_embeddings.shape[0]
        != len(unit_ids)
    ):

        raise ValueError(
            "Number of IDs does not match "
            "number of embedding rows."
        )


    # -----------------------------------------
    # Euclidean displacement
    # -----------------------------------------

    euclidean_distance = (
        np.linalg.norm(
            defective_embeddings
            -
            clean_embeddings,
            axis=1
        )
    )


    # -----------------------------------------
    # Cosine similarity
    # -----------------------------------------

    clean_norms = (
        np.linalg.norm(
            clean_embeddings,
            axis=1
        )
    )

    defective_norms = (
        np.linalg.norm(
            defective_embeddings,
            axis=1
        )
    )


    if (
        np.any(clean_norms == 0)
        or
        np.any(defective_norms == 0)
    ):

        raise ValueError(
            "Zero-norm embedding detected."
        )


    cosine_similarity = (
        np.sum(
            clean_embeddings
            *
            defective_embeddings,
            axis=1
        )
        /
        (
            clean_norms
            *
            defective_norms
        )
    )


    cosine_similarity = (
        np.clip(
            cosine_similarity,
            -1.0,
            1.0
        )
    )


    result = pd.DataFrame({
        "unit_id":
            unit_ids,

        "euclidean_distance":
            euclidean_distance,

        "cosine_similarity":
            cosine_similarity,

        "cosine_distance":
            1.0
            -
            cosine_similarity,

        "zero_displacement":
            euclidean_distance
            <= tolerance
    })


    if affected_ids is not None:

        affected_ids = set(
            affected_ids
        )

        result[
            "affected"
        ] = result[
            "unit_id"
        ].isin(
            affected_ids
        )


    return result

def summarize_embedding_distance_table(
    distance_df,
    affected_only=False
):
    """
    Summarize pair-level embedding diagnostics.
    """

    import numpy as np


    if (
        affected_only
        and
        "affected"
        not in distance_df.columns
    ):

        raise ValueError(
            "Affected labels are required "
            "for affected-only summaries."
        )


    if affected_only:

        df = distance_df[
            distance_df[
                "affected"
            ]
        ].copy()

    else:

        df = distance_df.copy()


    distances = (
        df[
            "euclidean_distance"
        ].to_numpy()
    )


    return {
        "n_pairs":
            len(df),

        "mean_distance":
            float(
                np.mean(
                    distances
                )
            ),

        "median_distance":
            float(
                np.median(
                    distances
                )
            ),

        "std_distance":
            float(
                np.std(
                    distances,
                    ddof=0
                )
            ),

        "q25_distance":
            float(
                np.quantile(
                    distances,
                    0.25
                )
            ),

        "q75_distance":
            float(
                np.quantile(
                    distances,
                    0.75
                )
            ),

        "max_distance":
            float(
                np.max(
                    distances
                )
            ),

        "mean_cosine_similarity":
            float(
                df[
                    "cosine_similarity"
                ].mean()
            ),

        "zero_displacement_count":
            int(
                df[
                    "zero_displacement"
                ].sum()
            ),

        "zero_displacement_rate":
            float(
                df[
                    "zero_displacement"
                ].mean()
            )
    }

def structural_campaign_summary(df):
    import numpy as np,pandas as pd
    rows=[]
    for campaign_id,g in df.groupby("campaign_id",sort=False):
        g=g.sort_values("message_order").copy()
        ts=pd.to_datetime(g["timestamp"])
        gaps=ts.diff().dt.total_seconds().div(60).dropna().to_numpy()
        channels=g["channel"].astype(str).to_numpy()
        lengths=g["text"].astype(str).str.len().to_numpy()
        tokens=g["text"].astype(str).str.split().str.len().to_numpy()
        counts=g["channel"].value_counts(normalize=True).to_numpy()
        entropy=0.0 if len(counts)<=1 else float(-(counts*np.log(counts)).sum()/np.log(len(counts)))
        switches=0.0 if len(channels)<=1 else float(np.mean(channels[1:]!=channels[:-1]))
        rows.append({"campaign_id":campaign_id,"n_messages":float(len(g)),"campaign_duration":float((ts.iloc[-1]-ts.iloc[0]).total_seconds()/60),"mean_time_gap":float(np.mean(gaps)),"median_time_gap":float(np.median(gaps)),"std_time_gap":float(np.std(gaps)),"n_channels":float(g["channel"].nunique()),"channel_switch_rate":switches,"channel_entropy":entropy,"dominant_channel_share":float(counts.max()),"mean_message_length":float(np.mean(lengths)),"std_message_length":float(np.std(lengths)),"mean_token_count":float(np.mean(tokens)),"std_token_count":float(np.std(tokens))})
    return pd.DataFrame(rows)