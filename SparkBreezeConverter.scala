package org.apache.spark.ml.classification

import breeze.linalg.{DenseVector => DenseBreezeVector, SparseVector => SparseBreezeVector, Vector => BreezeVector}
import org.apache.spark.ml.linalg.{DenseVector => DenseSparkVector, SparseVector => SparseSparkVector, Vector => SparkVector}

package object SparkBreezeConverter {

  implicit class RichSparkVector[I <: SparkVector](vector: I) {
    def asBreeze[O <: BreezeVector[Double]](implicit converter: Spark2BreezeConverter[I, O]): O = {
      converter.convert(vector)
    }
  }

  implicit class RichBreezeVector[I <: BreezeVector[Double]](breezeVector: I) {
    def fromBreeze[O <: SparkVector](implicit converter: Breeze2SparkConverter[I, O]): O = {
      converter.convert(breezeVector)
    }
  }
}

trait Spark2BreezeConverter[I <: SparkVector, O <: BreezeVector[Double]] {
  def convert(sparkVector: I): O
}

object Spark2BreezeConverter {
  implicit val denseSpark2DenseBreezeConverter: Spark2BreezeConverter[DenseSparkVector, DenseBreezeVector[Double]] = new Spark2BreezeConverter[DenseSparkVector, DenseBreezeVector[Double]] {
    override def convert(sparkVector: DenseSparkVector): DenseBreezeVector[Double] = {
      new DenseBreezeVector[Double](sparkVector.values)
    }
  }

  implicit val sparseSpark2SparseBreezeConverter: Spark2BreezeConverter[SparseSparkVector, SparseBreezeVector[Double]] = new Spark2BreezeConverter[SparseSparkVector, SparseBreezeVector[Double]] {
    override def convert(sparkVector: SparseSparkVector): SparseBreezeVector[Double] = {
      new SparseBreezeVector[Double](sparkVector.indices, sparkVector.values, sparkVector.size)
    }
  }

  implicit val defaultSpark2BreezeConverter: Spark2BreezeConverter[SparkVector, BreezeVector[Double]] = new Spark2BreezeConverter[SparkVector, BreezeVector[Double]] {
    override def convert(sparkVector: SparkVector): BreezeVector[Double] = {
      sparkVector match {
        case dv: DenseSparkVector => denseSpark2DenseBreezeConverter.convert(dv)
        case sv: SparseSparkVector => sparseSpark2SparseBreezeConverter.convert(sv)
      }
    }
  }
}

trait Breeze2SparkConverter[I <: BreezeVector[Double], O <: SparkVector] {
  def convert(breezeVector: I): O
}

object Breeze2SparkConverter {
  implicit val denseBreeze2DenseSparkVector: Breeze2SparkConverter[DenseBreezeVector[Double], DenseSparkVector] = new Breeze2SparkConverter[DenseBreezeVector[Double], DenseSparkVector] {
    override def convert(breezeVector: DenseBreezeVector[Double]): DenseSparkVector = {
      new DenseSparkVector(breezeVector.data)
    }
  }

  implicit val sparseBreeze2SparseSparkVector: Breeze2SparkConverter[SparseBreezeVector[Double], SparseSparkVector] = new Breeze2SparkConverter[SparseBreezeVector[Double], SparseSparkVector] {
    override def convert(breezeVector: SparseBreezeVector[Double]): SparseSparkVector = {
      val size = breezeVector.length
      val indices = breezeVector.array.index.take(size)
      val data = breezeVector.data.take(size)
      new SparseSparkVector(size, indices, data)
    }
  }

  implicit val defaultBreeze2SparkVector: Breeze2SparkConverter[BreezeVector[Double], SparkVector] = new Breeze2SparkConverter[BreezeVector[Double], SparkVector] {
    override def convert(breezeVector: BreezeVector[Double]): SparkVector = {
      breezeVector match {
        case dv: DenseBreezeVector[Double] => denseBreeze2DenseSparkVector.convert(dv)
        case sv: SparseBreezeVector[Double] => sparseBreeze2SparseSparkVector.convert(sv)
      }
    }
  }
}