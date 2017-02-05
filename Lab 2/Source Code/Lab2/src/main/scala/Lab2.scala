import org.apache.spark.{SparkContext, SparkConf}
/**
  * Created by Gulnoza on 2/5/2017.
  */
object Lab2 {

  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "C:\\winutils");

    val sparkConf = new SparkConf().setAppName("Lab2").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val input = sc.textFile("input")
    val output = input.flatMap(line => line.split(" ")).keyBy(n => n.charAt(0))
    output.saveAsTextFile("output")
    val o = output.collect()
    o.foreach {
      println
    }

  }
}
