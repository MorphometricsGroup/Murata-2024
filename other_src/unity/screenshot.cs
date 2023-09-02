using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Text;


public class screenshot : MonoBehaviour
{
    //private List<UnityEngine.GameObject> camera_list;
    //Camera[] camera_list;
    //private StreamWriter sw;

    // Start is called before the first frame update
    
    void Start()
    {
        StartCoroutine ("Screenshots");
    }

    // Update is called once per frame
    void Update()
    {

    }

    IEnumerator Screenshots()
    {
        int count;
        GameObject objected_1 = GameObject.Find("looked_1");
        GameObject objected_2 = GameObject.Find("looked_2");
        GameObject plant = GameObject.Find("one_leaf");
        List<float> listX;
        List<float> listZ;
        List<float> listY;
        (listX, listZ, listY) = CSVread();
        int listCount = listX.Count;
        count = Camera.allCamerasCount;
        GameObject obj = (GameObject)Resources.Load("Camera");
        Debug.Log(listCount.ToString());
        Material mat_black = (Material)Resources.Load("leaf_black");

        if(count < listCount)
        {
            while(count < listCount)
            {
                Instantiate(obj, new Vector3(0.0f, 2.0f, 0.0f), Quaternion.identity);
                count = Camera.allCamerasCount;
            }
        }

        Camera[] camera_list = new Camera[Camera.allCamerasCount];
        Camera.GetAllCameras(camera_list);

        for (int i = 0; i<count;i++)//i<?をカメラの個数に
        {
            // カメラ移動
            camera_list[i].gameObject.transform.position = new Vector3(listX[i], listY[i], listZ[i]);
            if(listZ[i]==1.25){
                camera_list[i].gameObject.transform.LookAt(objected_2.transform);
            }
            else{
                camera_list[i].gameObject.transform.LookAt(objected_1.transform);
            }
            camera_list[i].gameObject.name = "camera" + i;
            // 投影行列を取得
            Matrix4x4 v = camera_list[i].worldToCameraMatrix;
            
            StreamWriter sw = new StreamWriter("view_mat/" + i + @".csv", false, Encoding.GetEncoding("Shift_JIS"));
            string s1 = string.Join(",", v.ToString());
            sw.WriteLine(s1);
            sw.Flush();
	        sw.Close();
        }

        for(int s=0; s<8; s++)
        {
            int t = 0;
            foreach(Transform child in plant.transform)
                {
                    GameObject childObject = child.gameObject;
                    if (s==t)
                    {
                        //マテリアルsを設定
                        String leaf_name = String.Format("leaf_{0}",s);
                        Material mat_x = (Material)Resources.Load(leaf_name);
                        childObject.GetComponent<Renderer>().material = mat_x;
                    }

                    else
                    {
                        //マテリアル黒を設定
                        childObject.GetComponent<Renderer>().material = mat_black;
                    }
                    t += 1;
                }
            for (int i = 0; i<count; i++)//i<?をカメラの個数に
            {
                for (int j = 0; j<count; j++)//j<?をカメラの個数に
                {
                    if (i==j)
                    {
                        //Camera MainCamera = camera_list[j];
                        camera_list[j].enabled = true;
                    }
                    else
                    {
                        //Camera SubCamera = camera_list[j];
                        camera_list[j].enabled = false;
                    }
                }
                
                ScreenCapture.CaptureScreenshot(String.Format("image/{0}_{1}.png",i,s));
                yield return new WaitForSeconds(1f);
            }

        }

        
        
        Debug.Log("Done");
        
    }
    static (List<float> listX, List<float> listZ, List<float> listY) CSVread()
    {
        using (var reader = new StreamReader(@"Assets\Scripts\csv\quan_equidistant\cam_pos_64_equidistant.csv"))
        {
            List<string> listA = new List<string>();
            List<string> listB = new List<string>();
            List<string> listC = new List<string>();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');

                listA.Add(values[0]);
                listB.Add(values[1]);
                listC.Add(values[2]);
            }
            List<float> listX = listToint(listA);
            List<float> listZ = listToint(listB);
            List<float> listY = listToint(listC);
            return (listX, listZ, listY);
        }
        

    }
    static List<float> listToint(List<string> args)
    {
        List<float> list = new List<float>();
        foreach (String str in args)
        {
            float ret = float.Parse(str);
            list.Add(ret);
        }
        return list;
    }
}
