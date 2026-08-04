from src.export import export_to_parquet

import pyarrow.parquet as pq

def test_export_to_parquet_creates_file(tmp_path, fake_operations):

    output_file = tmp_path / "operations.parquet"

    export_to_parquet(operations= fake_operations, output_file=output_file)

    assert output_file.exists()

def test_export_to_parquet_has_correct_dimensions(tmp_path, fake_operations):

    output_file = tmp_path / "operations.parquet"

    export_to_parquet(operations= fake_operations, output_file=output_file)

    table = pq.read_table(output_file)

    assert table.num_rows == len(fake_operations)
    assert table.num_columns == len(fake_operations[0])

def test_export_to_parquet_has_correct_columns(
    tmp_path,
    fake_operations,
):
    output_file = tmp_path / "operations.parquet"

    export_to_parquet(
        operations=fake_operations,
        output_file=output_file,
    )

    table = pq.read_table(output_file)

    assert table.column_names == [
        "service",
        "service_name",
        "api_version",
        "operation",
        "description",
    ]

def test_export_to_parquet_contains_correct_data(
    tmp_path,
    fake_operations,
):
    output_file = tmp_path / "operations.parquet"

    export_to_parquet(
        operations=fake_operations,
        output_file=output_file,
    )

    table = pq.read_table(output_file)

    assert table.to_pylist() == fake_operations

def test_export_to_parquet_creates_parent_directories(
    tmp_path,
    fake_operations,
):
    output_file = (
        tmp_path
        / "data"
        / "raw"
        / "operations.parquet"
    )

    export_to_parquet(
        operations=fake_operations,
        output_file=output_file,
    )

    assert output_file.exists()