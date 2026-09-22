"""Physical matrix layouts shared by native-region lowering and its harness."""

MATRIX_LAYOUTS = ('contiguous', 'transposed', 'strided', 'offset',
                  'broadcast_rows', 'broadcast_cols')


def matrix_layout(rows, cols, layout):
    """Return element strides, base offset and storage size including a guard."""
    if rows < 1 or cols < 1:
        raise ValueError('Matrix dimensions must be positive')
    if layout == 'contiguous':
        stride_row, stride_col, offset = cols, 1, 0
    elif layout == 'transposed':
        stride_row, stride_col, offset = 1, rows, 0
    elif layout == 'strided':
        stride_row, stride_col, offset = 2 * cols + 3, 2, 0
    elif layout == 'offset':
        stride_row, stride_col, offset = cols + 3, 1, 7
    elif layout == 'broadcast_rows':
        stride_row, stride_col, offset = 0, 1, 0
    elif layout == 'broadcast_cols':
        stride_row, stride_col, offset = 1, 0, 0
    else:
        raise ValueError('Unsupported matrix layout: ' + layout)
    size = offset + (rows - 1) * stride_row + (cols - 1) * stride_col + 17
    return stride_row, stride_col, offset, size
