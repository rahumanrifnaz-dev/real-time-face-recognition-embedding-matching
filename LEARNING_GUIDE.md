# Learning Guide

## From pixels to a decision

**Face detection** finds where faces are in an image and returns boxes. **Face recognition** asks whose enrolled face an embedding resembles. Detection can succeed while recognition fails; they are separate steps.

A **face embedding** is a compact numeric vector produced by a neural network. During training, the model learned to place images of similar faces closer in this vector space and different faces farther apart. An embedding is not a name. The name comes from comparing it with locally labeled enrollment embeddings.

This project uses **Euclidean distance**, the straight-line distance between two vectors:

```text
distance(a, b) = sqrt(sum((a_i - b_i)^2))
```

Lower means more similar. **Cosine similarity** is another common metric; it compares vector directions, usually producing a larger value for more similar vectors. This project does not use cosine similarity, so its threshold values are not interchangeable with cosine thresholds.

## Thresholds and open-set recognition

A recognition threshold is the maximum Euclidean distance accepted as a known match. The nearest enrolled identity is not necessarily the correct identity: even a stranger has a mathematically nearest vector. If the best distance exceeds the threshold, the output is `Unknown`.

`Unknown` means “not sufficiently similar to any enrolled reference under this model and threshold.” It does not reveal who the person is and does not prove that they are absent in every possible image.

**Closed-set recognition** assumes every input belongs to one of the known classes and always chooses one. **Open-set recognition** allows an unknown result. This project is open-set.

A **false acceptance** happens when an unknown person is incorrectly labeled as enrolled. A **false rejection** happens when an enrolled person is labeled `Unknown`. A lenient (higher) distance threshold tends to increase false acceptance; a strict (lower) threshold tends to increase false rejection. Choose a threshold from held-out, consented observations rather than the enrollment images themselves. This trade-off is one reason the project is not a security-grade biometric authenticator.

## Why conditions matter

- Lighting changes contrast, shadows, highlights, and the pixels seen by both detector and encoder.
- Pose changes which facial structure is visible and can make alignment harder.
- Distance makes the face occupy fewer pixels, removing useful detail.
- Blur and occlusion also remove or change evidence. Do not label a failure's cause without a controlled comparison.

Multiple enrollment images expose the reference set to modest natural variation. They should be clear, mostly frontal, and include only moderate left/right views; poor or extreme samples can make matching less reliable.

## Real-time performance

**Inference latency** is the time spent processing an input. This project reports milliseconds measured with `time.perf_counter()`. **FPS** is frames displayed per second. They are related but not identical because recognition may run only every Nth frame.

Resizing a frame reduces the number of pixels the detector processes, often reducing latency, but it also shrinks faces and may hurt recognition at a distance. Processing every second frame reduces compute demand; the previous labels remain visible between processed frames, so fast motion may make boxes temporarily stale. Both settings should be experimentally compared.

## Person retrieval vs face recognition

Face recognition asks, "Who is this detected face?" and compares that face with
the complete local database. Person retrieval asks, "Where are these requested,
enrolled identities in the group image?"

Retrieval is a spatial search over every detected face. The system keeps each
detection's bounding box aligned with its embedding and compares every encodable
face with each requested identity's enrolled references. The assignment maximizes
the number of valid requested people, then minimizes their combined distance, and
each detected face can be assigned only once. The existing threshold still decides
whether each requested person has a valid match. People without an assigned
below-threshold match are reported as not found and are not highlighted.

Detection and encoding failures matter separately: a visible person cannot be
retrieved if their face is not detected, and a detected box cannot be matched if
the encoder produces no embedding. A successful result localizes a local label; it
does not establish a person's real-world identity or search outside the enrollment
database.

## Privacy and limitations

Enrollment and test images must come only from informed, consenting participants. Raw images must not be pushed to GitHub. Embeddings also encode biometric characteristics and require protection even though they are not ordinary photographs. This project ignores both by default and performs no network upload.

The system has a small local reference set, an off-the-shelf model, qualitative rather than laboratory-controlled conditions, no liveness detection, and no formal demographic or security evaluation. Camera quality, enrollment quality, participant diversity, threshold choice, and hardware all affect results. It is a learning tool, not surveillance, access control, or proof of identity.
