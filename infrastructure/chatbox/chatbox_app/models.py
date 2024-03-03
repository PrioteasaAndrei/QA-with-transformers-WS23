from django.db import models


class Session(models.Model):
    class Meta:
        db_table = 'session'
    id = models.BigAutoField(primary_key=True)
    date = models.DateTimeField(auto_now_add=True)

