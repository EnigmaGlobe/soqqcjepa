# Push-T Minimal C-JEPA FRD

## Purpose
Define the smallest usable feature scope for exposing C-JEPA as a self-hosted Python inference service for Push-T.

The goal is to validate that a learned object-centric latent world model can be called from an external client, such as Unity or ML-Agents, without forcing an ONNX export or changing the model itself.

## Problem Statement
The repository already includes a Push-T planning path that relies on pre-extracted slots and a trained C-JEPA predictor. What is missing is a simple, explicit definition of the minimum required behavior for a Python server that receives observations, runs inference, and returns predictions to a client.

## Scope
In scope:
- Load pre-extracted Push-T slot representations.
- Load a C-JEPA predictor checkpoint.
- Serve model inference through a Python process.
- Accept observations from an external client over a simple local API.
- Return latent predictions or rollout slots to the caller.
- Reuse the existing Push-T benchmark setup for validation.
- Support multiple seeds for repeatability.

Out of scope:
- Training a new object-centric encoder.
- Training a new C-JEPA checkpoint from scratch.
- Exporting to ONNX.
- Adding a new Push-T environment or reward definition.
- Replacing the Python model with a non-Python runtime.
- Changing Stable-WorldModel internals.

## 4-Phase MVP Plan
### Phase 1: Local Model Run
Goal: prove the model can run locally in Python without any client or server wrapper.

What ships:
- A local script or notebook entry point that loads a Push-T C-JEPA checkpoint.
- A single observation payload from disk or an in-memory test fixture.
- A forward pass that returns latent predictions for that one payload.
- A minimal text or console printout showing the output shape or summary.
- Implementation target: `scripts/pusht/phase1_local_smoke.py`.

Input:
- Push-T slot features or a small pre-extracted slot sample.

Output:
- One valid latent prediction tensor or rollout slot tensor.

Exit criteria:
- The model loads successfully in Python.
- One local request returns a valid output.
- The output shape is inspectable and matches the expected latent format.
- No client process or server process is required.

Not included in phase 1:
- Network transport.
- Unity or ML-Agents integration.
- Multi-request batching.
- Push-T benchmark scoring.

### Phase 2: Frame-Sequence-to-Slots Pipeline
Goal: prove a single Python main can take a short ordered frame sequence, extract slots with VideoSAUR, and feed those slots into the Push-T C-JEPA world model.

What ships:
- A Python entrypoint that loads 4-5 ordered frames from disk or a synthetic fixture.
- A VideoSAUR checkpoint that converts the frame sequence into latent slots.
- A C-JEPA world-model checkpoint that consumes those slots and predicts the next latent state.
- A small amount of slot shaping so the VideoSAUR output matches the world-model checkpoint contract.
- Implementation target: `scripts/pusht/phase2_client_integration_smoke.py`.

Exit criteria:
- The frame sequence is converted to latent slots through VideoSAUR.
- The world model consumes those slots and returns a prediction.
- The run prints the key tensor shapes for the frames, slots, and prediction.

### Phase 3: Push-T Evaluation Pass
Goal: prove the pipeline output is useful in the existing benchmark path.

Exit criteria:
- At least one seeded run completes.
- The evaluation writes a metrics file.
- The output is usable by the existing Push-T planning path.

### Phase 4: Python Server Packaging
Goal: wrap the already-working local path in a self-hosted Python service.

What ships:
- A local endpoint for observation requests.
- A thin transport layer around the local inference code.
- The same checkpoint and output behavior from Phase 1.

Exit criteria:
- The server starts from the same model code.
- The client can call the server without changing the model logic.

## Minimal User Flow
1. Prepare a short Push-T frame sequence or use a synthetic fixture.
2. Choose a VideoSAUR checkpoint and a C-JEPA checkpoint for Push-T.
3. Run the Python pipeline main so the frame sequence becomes VideoSAUR slots.
4. Feed the slots into the Push-T world-model checkpoint.
5. Receive the predicted latent output from the world model.
6. Run the existing Push-T evaluation path to confirm the output is usable.
7. Review the planning metrics written by the evaluation script.

## Functional Requirements
### FR1. Data Availability
The system must accept Push-T data in the Stable-WorldModel format described in `docs/DATASET.md`.

### FR2. Checkpoint Selection
The system must allow the user to point the Python server at a specific C-JEPA checkpoint.

### FR3. Server Inference
The system must expose a Python endpoint that accepts a Push-T observation payload and returns a model output.

### FR4. Client Integration
The system must allow an external client to call the server locally during a simulated Push-T run.

### FR5. Seeded Evaluation
The system must support running the evaluation over multiple seeds for basic robustness checks.

### FR6. Result Capture
The system must write the planning or evaluation results to a text output file so runs can be compared later.

## Acceptance Criteria
The minimal feature is considered working when all of the following are true:
- Phase 1 completes with a working local model run and validated output shape.
- Phase 2 completes with a working frame-sequence integration.
- Phase 3 completes with at least one seeded Push-T evaluation run.
- Phase 4 packages the same logic into a Python server.
- The run produces a metrics text file in the repository output path.

## Dependencies
- Stable-WorldModel
- Stable-Pretraining
- A Push-T dataset in the expected cache format
- A C-JEPA checkpoint trained for Push-T
- A local Python runtime that can host the inference service
- A client process that can call the Python server

## Risks / Notes
- The repo does not ship a local checkpoint by default, so the user must download one.
- The repo uses third-party planning code, so failures may come from external package versions rather than C-JEPA itself.
- A Python server is easier to integrate than ONNX, but it still needs a thin client/server contract.
- Push-T planning is a downstream integration test, not proof of general intelligence.

## References
- `README.md`
- `docs/DATASET.md`
- `scripts/pusht/test_planning.sh`
- `src/plan/run.py`
