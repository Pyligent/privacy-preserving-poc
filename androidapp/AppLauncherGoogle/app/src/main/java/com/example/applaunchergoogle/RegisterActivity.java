package com.example.applaunchergoogle;

import androidx.appcompat.app.AppCompatActivity;

import android.content.ContentValues;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;

public class RegisterActivity extends AppCompatActivity {
    EditText username, password, email, country, dob;
    RadioGroup gender;
    Button register, cancel;
    DatabaseConnector databaseConn = new DatabaseConnector(this);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        username = findViewById(R.id.username);
        password = findViewById(R.id.password);
        email = findViewById(R.id.email);
        country = findViewById(R.id.country);
        dob = findViewById(R.id.dob);
        gender = findViewById(R.id.gender);
        register = findViewById(R.id.register);
        cancel = findViewById(R.id.cancel);

        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String usernameValue = username.getText().toString();
                String passValue = password.getText().toString();
                String emailvalue = username.getText().toString();
                String countryValue = password.getText().toString();
                String dobValue = username.getText().toString();
                RadioButton checkBtn = findViewById(gender.getCheckedRadioButtonId());
                String genderValue = checkBtn.getText().toString();


                if(usernameValue.length()>1){
                    ContentValues contentValues = new ContentValues();
                    contentValues.put("username",usernameValue);
                    contentValues.put("password",passValue);
                    contentValues.put("email",emailvalue);
                    contentValues.put("country",countryValue);
                    contentValues.put("dob",dobValue);
                    contentValues.put("gender",genderValue);

                    databaseConn.insertNewUser(contentValues);
                    Toast.makeText(RegisterActivity.this, "User is registered!", Toast.LENGTH_SHORT).show();
                }
                else{
                    Toast.makeText(RegisterActivity.this, "Enter user information", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}
