using System.Buffers;
using System.Collections.Immutable;

namespace zms9110750.ReedSolomon.Matrixs;
/// <summary>
/// 矩阵扩展方法，提供解码（分片恢复）功能
/// </summary>
public static class MatrixExtensions
{
    /// <summary>
    /// 使用连续内存布局执行解码：从任意 K 个可用分片恢复全部 K 个数据分片
    /// </summary>
    /// <param name="encodingMatrix">编码矩阵（原始矩阵）</param>
    /// <param name="availableShards">K 个可用分片连续拼接，长度为 Columns × blockSize</param>
    /// <param name="recoveredDataShards">K 个数据分片连续拼接，长度为 Columns × blockSize，会被写入</param>
    /// <param name="availableRowIndices">可用分片对应的矩阵行索引，长度为 Columns</param>
    /// <param name="blockSize">每个分片的字节数</param>
    /// <exception cref="ArgumentException">当 availableRowIndices 长度不等于 Columns 时抛出</exception>
    /// <exception cref="ArgumentException">当 availableShards 或 recoveredDataShards 长度不等于 Columns × blockSize 时抛出</exception>
    /// <remarks>多次使用，应该自行求逆矩阵并缓存，避免重复计算。
    /// <code>
    /// var inverse = encodingMatrix.InverseRows(availableRowIndices, dataShardCount);
    /// inverse.CodeShards(availableShards, recoveredDataShards, blockSize);
    /// </code>
    /// </remarks>
    public static void RecoverDataShards(
        this IMatrix<byte> encodingMatrix,
        ReadOnlySpan<byte> availableShards,
        Span<byte> recoveredDataShards,
        ReadOnlySpan<int> availableRowIndices,
        int blockSize)
    {
        // 验证 blockSize 必须大于 0
        if (blockSize <= 0)
        {
            throw new ArgumentException("blockSize 必须大于 0", nameof(blockSize));
        }
        int dataShardCount = encodingMatrix.Columns;

        // 验证 availableRowIndices 长度必须等于数据分片数
        if (availableRowIndices.Length != dataShardCount)
        {
            throw new ArgumentException($"availableRowIndices 长度应为 {dataShardCount}，实际 {availableRowIndices.Length}", nameof(availableRowIndices));
        }

        // 验证 availableShards 长度
        int expectedInputLength = dataShardCount * blockSize;
        if (availableShards.Length != expectedInputLength)
        {
            throw new ArgumentException($"availableShards 长度应为 {expectedInputLength}，实际 {availableShards.Length}", nameof(availableShards));
        }

        // 验证 recoveredDataShards 长度
        int expectedOutputLength = dataShardCount * blockSize;
        if (recoveredDataShards.Length != expectedOutputLength)
        {
            throw new ArgumentException($"recoveredDataShards 长度应为 {expectedOutputLength}，实际 {recoveredDataShards.Length}", nameof(recoveredDataShards));
        }

        // 求逆矩阵并执行解码
        var inverse = encodingMatrix.InverseRows(availableRowIndices);
        inverse.CodeShards(availableShards, recoveredDataShards, blockSize);
    }

    /// <summary>
    /// 使用分片集合执行解码：恢复缺失的分片
    /// </summary>
    /// <param name="encodingMatrix">编码矩阵（原始矩阵）</param>
    /// <param name="availableShards">可用的输入分片，数量为 K</param>
    /// <param name="recoveredDataShards">恢复出的数据分片，数量为 K。会被写入</param>
    /// <param name="availableRowIndices">可用分片对应的矩阵行索引，长度为 K</param>
    /// <remarks>多次使用，应该自行求逆矩阵并缓存，避免重复计算。
    /// <code>
    /// var inverse = encodingMatrix.InverseRows(availableRowIndices, dataShardCount);
    /// inverse.CodeShards(availableShards, recoveredDataShards);
    /// </code>
    /// </remarks>
    /// <exception cref="ArgumentNullException">当参数为 null 时抛出</exception>
    /// <exception cref="ArgumentException">当分片数量或长度不正确时抛出</exception>
    public static void RecoverDataShards(
        this IMatrix<byte> encodingMatrix,
        ReadOnlyMemory<ReadOnlyMemory<byte>> availableShards,
        ReadOnlyMemory<Memory<byte>> recoveredDataShards,
        ReadOnlySpan<int> availableRowIndices)
    {
        int dataShardCount = encodingMatrix.Columns;

        // 验证参数非空
        if (availableShards.IsEmpty)
        {
            throw new ArgumentNullException(nameof(availableShards));
        }
        if (recoveredDataShards.IsEmpty)
        {
            throw new ArgumentNullException(nameof(recoveredDataShards));
        }
        if (availableRowIndices.IsEmpty)
        {
            throw new ArgumentNullException(nameof(availableRowIndices));
        }

        // 验证 availableRowIndices 长度必须等于数据分片数
        if (availableRowIndices.Length != dataShardCount)
        {
            throw new ArgumentException($"availableRowIndices 长度应为 {dataShardCount}，实际 {availableRowIndices.Length}", nameof(availableRowIndices));
        }

        // 验证可用分片数量
        if (availableShards.Length != dataShardCount)
        {
            throw new ArgumentException($"availableShards 数量应为 {dataShardCount}，实际 {availableShards.Length}", nameof(availableShards));
        }

        // 验证恢复分片数量
        if (recoveredDataShards.Length != dataShardCount)
        {
            throw new ArgumentException($"recoveredDataShards 数量应为 {dataShardCount}，实际 {recoveredDataShards.Length}", nameof(recoveredDataShards));
        }

        // 获取第一个分片的长度作为基准
        int length = availableShards.Span[0].Length;

        // 验证所有可用分片长度一致
        for (int i = 1; i < dataShardCount; i++)
        {
            if (availableShards.Span[i].Length != length)
            {
                throw new ArgumentException($"可用分片长度不一致：分片0长度为 {length}，分片{i}长度为 {availableShards.Span[i].Length}");
            }
        }

        // 验证所有恢复分片长度一致
        for (int i = 0; i < dataShardCount; i++)
        {
            if (recoveredDataShards.Span[i].Length != length)
            {
                throw new ArgumentException($"恢复分片{i}长度应为 {length}，实际 {recoveredDataShards.Span[i].Length}");
            }
        }

        // 求逆矩阵并执行编码
        var inverse = encodingMatrix.InverseRows(availableRowIndices);
        inverse.CodeShards(availableShards, recoveredDataShards);
    }

    /// <summary>
    /// 使用分片集合执行解码（兼容版本）
    /// </summary>
    /// <param name="encodingMatrix">编码矩阵（原始矩阵）</param>
    /// <param name="availableShards">可用的输入分片，数量为 K</param>
    /// <param name="recoveredDataShards">恢复出的数据分片，数量为 K。会被写入</param>
    /// <param name="availableRowIndices">可用分片对应的矩阵行索引，长度为 K</param>
    /// <param name="offset">每个分片的起始字节索引</param>
    /// <param name="count">每个分片要处理的字节数</param>
    /// <remarks>多次使用，应该自行求逆矩阵并缓存，避免重复计算。
    /// <code>
    /// var inverse = encodingMatrix.InverseRows(availableRowIndices, dataShardCount);
    /// inverse.CodeShards(availableShards, recoveredDataShards, offset, count);
    /// </code>
    /// </remarks>
    /// <exception cref="ArgumentNullException">当参数为 null 时抛出</exception>
    /// <exception cref="ArgumentException">当分片数量不足或长度不足时抛出</exception>
    public static void RecoverDataShards(
        this IMatrix<byte> encodingMatrix,
        IEnumerable<IList<byte>> availableShards,
        IEnumerable<IList<byte>> recoveredDataShards,
        ReadOnlySpan<int> availableRowIndices,
        int offset,
        int count)
    {

        // 验证 offset 和 count
        if (offset < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), "offset 不能为负数");
        }
        if (count <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "count 必须大于 0");
        }

        int dataShardCount = encodingMatrix.Columns;

        // 验证 availableRowIndices 长度必须等于数据分片数
        if (availableRowIndices.Length != dataShardCount)
        {
            throw new ArgumentException($"availableRowIndices 长度应为 {dataShardCount}，实际 {availableRowIndices.Length}", nameof(availableRowIndices));
        }

        // 将可用分片转为列表并验证数量
        var availableList = availableShards as IReadOnlyList<IList<byte>> ?? availableShards?.ToImmutableList() ?? throw new ArgumentNullException(nameof(availableShards));
        if (availableList.Count != dataShardCount)
        {
            throw new ArgumentException($"availableShards 数量应为 {dataShardCount}，实际 {availableList.Count}", nameof(availableShards));
        }

        // 将缺失分片转为列表并验证数量
        var missingList = recoveredDataShards as IReadOnlyList<IList<byte>> ?? recoveredDataShards?.ToImmutableList() ?? throw new ArgumentNullException(nameof(recoveredDataShards));
        if (missingList.Count != dataShardCount)
        {
            throw new ArgumentException(
                $"recoveredDataShards 数量应为 {dataShardCount}，实际 {missingList.Count}",
                nameof(recoveredDataShards));
        }

        // 验证所有可用分片长度足够
        for (int i = 0; i < dataShardCount; i++)
        {
            if (availableList[i].Count < offset + count)
            {
                throw new ArgumentException($"可用分片 {i} 长度不足，需要 {offset + count}，实际 {availableList[i].Count}");
            }
        }

        // 验证所有缺失分片长度足够
        for (int i = 0; i < dataShardCount; i++)
        {
            if (missingList[i].Count < offset + count)
            {
                throw new ArgumentException($"缺失分片 {i} 长度不足，需要 {offset + count}，实际 {missingList[i].Count}");
            }
        }

        var inverse = encodingMatrix.InverseRows(availableRowIndices);
        inverse.CodeShards(availableShards, recoveredDataShards, offset, count);
    }

    // ===== 以下为 IMatrix.CodeShards 的扩展方法重载 =====

    /// <summary>
    /// 使用分片集合执行矩阵乘法（扩展方法）。内部将分片展平为连续内存后调用 Span 版本。
    /// </summary>
    /// <param name="matrix">矩阵实例</param>
    /// <param name="inputs">Columns 个数据分片。所有分片长度必须相等。</param>
    /// <param name="outputs">输出分片。若矩阵非方阵：数量 Rows - Columns；若方阵：数量 Columns。</param>
    public static void CodeShards(
        this IMatrix matrix,
        ReadOnlyMemory<ReadOnlyMemory<byte>> inputs,
        ReadOnlyMemory<Memory<byte>> outputs)
    {
        int columns = matrix.Columns;
        int expectedOutputCount = matrix.IsSquare ? matrix.Rows : (matrix.Rows - matrix.Columns);

        // 验证
        if (inputs.IsEmpty) throw new ArgumentNullException(nameof(inputs));
        if (outputs.IsEmpty) throw new ArgumentNullException(nameof(outputs));
        if (inputs.Length != columns)
            throw new ArgumentException($"输入分片数量应为 {columns}，实际 {inputs.Length}", nameof(inputs));
        if (outputs.Length != expectedOutputCount)
            throw new ArgumentException($"输出分片数量应为 {expectedOutputCount}，实际 {outputs.Length}", nameof(outputs));

        int length = inputs.Span[0].Length;
        for (int i = 1; i < columns; i++)
        {
            if (inputs.Span[i].Length != length)
                throw new ArgumentException($"输入分片长度不一致：分片0长度为 {length}，分片{i}长度为 {inputs.Span[i].Length}");
        }
        for (int i = 0; i < expectedOutputCount; i++)
        {
            if (outputs.Span[i].Length != length)
                throw new ArgumentException($"输出分片{i}长度应为 {length}，实际 {outputs.Span[i].Length}");
        }

        // 展平到连续缓冲区
        int blockSize = length;
        byte[] buffer = ArrayPool<byte>.Shared.Rent(columns * blockSize + expectedOutputCount * blockSize);
        try
        {
            Span<byte> flatInputs = buffer.AsSpan(0, columns * blockSize);
            Span<byte> flatOutputs = buffer.AsSpan(columns * blockSize, expectedOutputCount * blockSize);

            for (int col = 0; col < columns; col++)
            {
                inputs.Span[col].Span.CopyTo(flatInputs.Slice(col * blockSize, blockSize));
            }

            matrix.CodeShards(flatInputs, flatOutputs, blockSize);

            for (int row = 0; row < expectedOutputCount; row++)
            {
                flatOutputs.Slice(row * blockSize, blockSize).CopyTo(outputs.Span[row].Span);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// 使用分片集合执行矩阵乘法（扩展方法）。内部将分片展平为连续内存后调用 Span 版本。
    /// </summary>
    /// <param name="matrix">矩阵实例</param>
    /// <param name="inputs">Columns 个数据分片</param>
    /// <param name="outputs">输出分片。若矩阵非方阵：数量 Rows - Columns；若方阵：数量 Columns。</param>
    /// <param name="offset">每个分片的起始字节索引</param>
    /// <param name="count">每个分片要处理的字节数</param>
    public static void CodeShards(
        this IMatrix matrix,
        IEnumerable<IList<byte>> inputs,
        IEnumerable<IList<byte>> outputs,
        int offset,
        int count)
    {
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));

        int columns = matrix.Columns;
        int expectedOutputCount = matrix.IsSquare ? matrix.Rows : (matrix.Rows - matrix.Columns);

        var inputList = inputs as IReadOnlyList<IList<byte>> ?? inputs?.ToImmutableList() ?? throw new ArgumentNullException(nameof(inputs));
        var outputList = outputs as IReadOnlyList<IList<byte>> ?? outputs?.ToImmutableList() ?? throw new ArgumentNullException(nameof(outputs));

        if (inputList.Count != columns)
            throw new ArgumentException($"输入分片数量应为 {columns}，实际 {inputList.Count}", nameof(inputs));
        if (outputList.Count != expectedOutputCount)
            throw new ArgumentException($"输出分片数量应为 {expectedOutputCount}，实际 {outputList.Count}", nameof(outputs));

        for (int col = 0; col < columns; col++)
        {
            if (inputList[col].Count < offset + count)
                throw new ArgumentException($"输入分片 {col} 长度不足，需要 {offset + count}，实际 {inputList[col].Count}");
        }
        for (int row = 0; row < expectedOutputCount; row++)
        {
            if (outputList[row].Count < offset + count)
                throw new ArgumentException($"输出分片 {row} 长度不足，需要 {offset + count}，实际 {outputList[row].Count}");
        }

        // 展平到连续缓冲区
        byte[] buffer = ArrayPool<byte>.Shared.Rent(columns * count + expectedOutputCount * count);
        try
        {
            Span<byte> flatInputs = buffer.AsSpan(0, columns * count);
            Span<byte> flatOutputs = buffer.AsSpan(columns * count, expectedOutputCount * count);

            for (int col = 0; col < columns; col++)
            {
                inputList[col].AsReadOnlySpan(offset, count).CopyTo(flatInputs.Slice(col * count, count));
            }

            matrix.CodeShards(flatInputs, flatOutputs, count);

            for (int row = 0; row < expectedOutputCount; row++)
            {
                flatOutputs.Slice(row * count, count).CopyTo(outputList[row].AsSpan(offset, count));
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}