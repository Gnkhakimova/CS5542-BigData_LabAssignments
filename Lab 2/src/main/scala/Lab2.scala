import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
/**
  * Created by Gulnoza on 1/31/2017.
  */
object Lab2 {

  def main (args: Array[String])
  {
    val sparkConf = new SparkConf().setAppName("Lab2").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)
    val lines = List ("Hello world", "Hello Spark", "Welcome to Spark")
    val first = lines.map(line=>line.split("\\s+"))
    val second = first.groupBy(w=>w.charAt(0))
    val third = second.countByKey()
    dbutils.fs.rm("C:\\Users\\Gulnoza\\Documents\\UMKC2016\\CS5542\\CS5542-BigData_LabAssignments\\Lab 2", true)
    third.saveAsTextFile("C:\\Users\\Gulnoza\\Documents\\UMKC2016\\CS5542\\CS5542-BigData_LabAssignments\\Lab 2")
    val fourth = sc.textFile ("C:\\Users\\Gulnoza\\Documents\\UMKC2016\\CS5542\\CS5542-BigData_LabAssignments\\Lab 2")
    println(fourth.collect().mkString(", "))
  }
}
