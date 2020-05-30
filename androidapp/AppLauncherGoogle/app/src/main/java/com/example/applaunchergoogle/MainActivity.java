package com.example.applaunchergoogle;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button secondAct = (Button) findViewById(R.id.secondActButton);
        secondAct.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent secondActivity = new Intent(getApplicationContext(),SecondActivity.class);
                secondActivity.putExtra("com.example.applaunchergoogle.actikey","Welcome to the app!");
                startActivity(secondActivity);
            }
        });

        Button gsearch = (Button)findViewById(R.id.gsearchButton);
        gsearch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String text = "http://www.google.com";
                Uri address = Uri.parse(text);
                Intent tosearchgoogle = new Intent(Intent.ACTION_VIEW, address);
                if (tosearchgoogle.resolveActivity(getPackageManager()) != null){
                    startActivity(tosearchgoogle);
                }
            }
        });
    }
}
