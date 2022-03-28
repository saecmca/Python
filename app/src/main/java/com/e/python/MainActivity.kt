package com.e.python

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        val pyInstance = Python.getInstance()
        val pyModule = pyInstance.getModule("ProcessImage")
        val pyAttribute = pyModule.callAttr("test","Mani","  test")

        findViewById<TextView>(R.id.tv).text=pyAttribute.toString()
    }
}