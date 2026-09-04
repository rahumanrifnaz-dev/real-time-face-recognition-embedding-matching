"""Streamlit interface for consent-based local group-image retrieval."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from face_database import DEFAULT_DATABASE_PATH, DEFAULT_THRESHOLD, load_database
from face_detector import detect_faces
from face_encoder import get_face_embedding
from group_search import annotate_people
from person_retrieval import (
    IdentityNotEnrolledError,
    parse_person_queries,
    retrieve_people,
)
from utils import bgr_to_rgb


st.set_page_config(
    page_title="Group person search",
    page_icon=":material/person_search:",
    layout="wide",
)
st.title("Group image person search")
st.caption(
    "Upload a consented group image and search for one or more identities enrolled locally."
)
st.session_state.setdefault("group_search_multi_result", None)

database_path = Path(
    st.sidebar.text_input("Local database", value=str(DEFAULT_DATABASE_PATH))
)
threshold = st.sidebar.slider(
    "Maximum Euclidean distance", min_value=0.20, max_value=1.00,
    value=float(DEFAULT_THRESHOLD), step=0.01,
)
show_all_faces = st.sidebar.checkbox("Show all detected faces", value=True)

try:
    database = load_database(database_path)
except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

identities = sorted(name for name, references in database.items() if references)
if not identities:
    st.error("The local embedding database contains no enrolled identities.")
    st.stop()

st.write("Available local labels: " + ", ".join(f"`{name}`" for name in identities))
example_query = (
    f"Find {identities[0]} and {identities[1]}"
    if len(identities) > 1
    else f"Find {identities[0]}"
)
with st.form("group_person_search", border=False):
    query = st.text_input(
        "Who should be highlighted?",
        placeholder=f"For example: {example_query}",
    )
    uploaded_file = st.file_uploader(
        "Upload a group image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    submitted = st.form_submit_button("Find people", type="primary")

if submitted:
    st.session_state["group_search_multi_result"] = None
    try:
        targets = parse_person_queries(query, identities)
    except IdentityNotEnrolledError as exc:
        st.error(str(exc).capitalize() + ".")
    else:
        if uploaded_file is None:
            st.error("Upload a group image first.")
        else:
            encoded_file = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(encoded_file, cv2.IMREAD_COLOR)
            if frame is None:
                st.error("The uploaded image could not be opened.")
            else:
                try:
                    rgb = bgr_to_rgb(frame)
                    locations = detect_faces(rgb)
                    embeddings = [get_face_embedding(rgb, box) for box in locations]
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    if not locations:
                        st.warning("No faces were detected in the supplied image.")
                    else:
                        results = retrieve_people(
                            embeddings, locations, targets, database, threshold
                        )
                        annotated = annotate_people(
                            frame, locations, results, show_all_faces=show_all_faces
                        )
                        saved, png_buffer = cv2.imencode(".png", annotated)
                        st.session_state["group_search_multi_result"] = {
                            "image": bgr_to_rgb(annotated),
                            "png": png_buffer.tobytes() if saved else None,
                            "results": results,
                            "faces_detected": len(locations),
                            "faces_encoded": sum(item is not None for item in embeddings),
                        }

display_data = st.session_state.get("group_search_multi_result")
if display_data:
    results = display_data["results"]
    found_results = [result for result in results if result["found"]]
    missing_results = [result for result in results if not result["found"]]
    if found_results:
        found_names = ", ".join(result["person"] for result in found_results)
        st.success(f"Found: {found_names}.")
    if missing_results:
        missing_names = ", ".join(result["person"] for result in missing_results)
        st.warning(f"Not found within the current threshold: {missing_names}.")

    first, second, third = st.columns(3)
    first.metric("Faces detected", display_data["faces_detected"])
    second.metric("Faces encoded", display_data["faces_encoded"])
    third.metric("Targets found", f"{len(found_results)}/{len(results)}")

    st.subheader("Requested identities")
    for result in results:
        distance_text = (
            f"{result['distance']:.3f}"
            if np.isfinite(result["distance"])
            else "unavailable"
        )
        decision = "FOUND" if result["found"] else "NOT FOUND"
        st.write(f"- **{result['person']}** — {decision} — distance: `{distance_text}`")

    st.image(display_data["image"], caption="Group-search result", width="stretch")
    if display_data["png"] is not None:
        st.download_button(
            "Download annotated image",
            data=display_data["png"],
            file_name="group_search_multiple_people.png",
            mime="image/png",
        )

st.info(
    "The text parser only maps your words to local enrolled labels. It does not "
    "identify public figures, search the internet, or send the image to an LLM."
)
