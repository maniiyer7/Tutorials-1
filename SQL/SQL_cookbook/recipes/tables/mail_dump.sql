-- MySQL dump 10.13  Distrib 5.6.26, for osx10.8 (x86_64)
--
-- Host: localhost    Database: cookbook
-- ------------------------------------------------------
-- Server version	5.6.26

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `mail`
--

DROP TABLE IF EXISTS `mail`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mail` (
  `t` datetime DEFAULT NULL,
  `srcuser` varchar(8) DEFAULT NULL,
  `srchost` varchar(20) DEFAULT NULL,
  `dstuser` varchar(8) DEFAULT NULL,
  `dsthost` varchar(20) DEFAULT NULL,
  `size` bigint(20) DEFAULT NULL,
  KEY `t` (`t`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mail`
--

LOCK TABLES `mail` WRITE;
/*!40000 ALTER TABLE `mail` DISABLE KEYS */;
INSERT INTO `mail` VALUES ('2014-05-11 10:15:08','barb','saturn','tricia','mars',58274),('2014-05-12 12:48:13','tricia','mars','gene','venus',194925),('2014-05-12 15:02:49','phil','mars','phil','saturn',1048),('2014-05-12 18:59:18','barb','saturn','tricia','venus',271),('2014-05-14 09:31:37','gene','venus','barb','mars',2291),('2014-05-14 11:52:17','phil','mars','tricia','saturn',5781),('2014-05-14 14:42:21','barb','venus','barb','venus',98151),('2014-05-14 17:03:01','tricia','saturn','phil','venus',2394482),('2014-05-15 07:17:48','gene','mars','gene','saturn',3824),('2014-05-15 08:50:57','phil','venus','phil','venus',978),('2014-05-15 10:25:52','gene','mars','tricia','saturn',998532),('2014-05-15 17:35:31','gene','saturn','gene','mars',3856),('2014-05-16 09:00:28','gene','venus','barb','mars',613),('2014-05-16 23:04:19','phil','venus','barb','venus',10294),('2014-05-19 12:49:23','phil','mars','tricia','saturn',873),('2014-05-19 22:21:51','gene','saturn','gene','venus',23992);
/*!40000 ALTER TABLE `mail` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2015-11-04  5:59:21
