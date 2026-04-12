import pathlib
import sys

import pytest
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from runtime import ImageRequest, RequestTable, TensorRTRuntime  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the multistream runtime test",
)


def make_request(req_id: int, hw: int) -> ImageRequest:
    tensor = torch.randn(3, hw, hw, device="cuda", dtype=torch.float32)
    return ImageRequest(req_id=req_id, tensor=tensor)


def test_bucket_selection_and_output_shape():
    runtime = TensorRTRuntime()
    table = RequestTable(
        requests=[
            make_request(0, 224),
            make_request(1, 256),
            make_request(2, 512),
        ]
    )

    results = runtime.execute(table)
    buckets = {result.req_id: result.bucket_name for result in results}

    assert buckets[0] == "224"
    assert buckets[1] == "256"
    assert buckets[2] == "512"

    for result, hw in zip(results, (224, 256, 512)):
        assert result.output.shape == (1, 3, hw, hw)


def test_multistream_execution_uses_multiple_bucket_streams():
    runtime = TensorRTRuntime()
    table = RequestTable(
        requests=[
            make_request(0, 224),
            make_request(1, 224),
            make_request(2, 256),
            make_request(3, 512),
        ]
    )

    results = runtime.execute(table)
    stream_names = {result.stream_name for result in results}

    assert "stream_224" in stream_names
    assert "stream_256" in stream_names
    assert "stream_512" in stream_names
    assert len(stream_names) >= 3
