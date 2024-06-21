import uuid
from datetime import datetime, timedelta

import boto3
from botocore.config import Config
from django.conf import settings
from django.db import models
from django.utils import timezone
#from django_use_email_as_username.models import BaseUser, BaseUserManager
from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.utils.translation import gettext_lazy as _

from jose import jwt
from yarl import URL


class UUIDPrimaryKeyBase(models.Model):
    id = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)

    class Meta:
        abstract = True


class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(editable=False, auto_now_add=True)
    modified_at = models.DateTimeField(editable=False, auto_now=True)

    class Meta:
        abstract = True
        ordering = ["created_at"]


class RedboxUserManager(BaseUserManager):
    """Define a model manager for User model with no username field."""

    use_in_migrations = True

    def _create_user(self, username, password, **extra_fields):
        """Create and save a User with the given email and password."""
        if not username:
            raise ValueError("The given email must be set")
        #email = self.normalize_email(email)
        #user = self.model(email=email, **extra_fields)
        User.set_password(password)
        User.save(using=self._db)
        return user

    def create_user(self, username, password=None, **extra_fields):
        """Create and save a regular User with the given email and password."""
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(username, password, **extra_fields)

    def create_superuser(self, username, password, **extra_fields):
        """Create and save a SuperUser with the given email and password."""
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(username, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin, UUIDPrimaryKeyBase):
    username = models.EmailField(unique=True)
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=48)
    last_name = models.CharField(max_length=48)
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    date_joined = models.DateTimeField(default=timezone.now)


    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = []
    objects = RedboxUserManager()

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.email}"

    def save(self, *args, **kwargs):
        self.email = self.email.lower()
        super().save(*args, **kwargs)

    def get_bearer_token(self) -> str:
        """the bearer token expected by the core-api"""
        user_uuid = str(self.id)
        bearer_token = jwt.encode({"user_uuid": user_uuid}, key=settings.SECRET_KEY)
        return f"Bearer {bearer_token}"


class StatusEnum(models.TextChoices):
    uploaded = "uploaded"
    parsing = "parsing"
    chunking = "chunking"
    embedding = "embedding"
    indexing = "indexing"
    complete = "complete"
    unknown = "unknown"
    deleted = "deleted"
    errored = "errored"


class File(UUIDPrimaryKeyBase, TimeStampedModel):
    status = models.CharField(choices=StatusEnum.choices, null=False, blank=False)
    original_file = models.FileField(storage=settings.STORAGES["default"]["BACKEND"])
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_file_name = models.TextField(max_length=2048, blank=True, null=True)
    core_file_uuid = models.UUIDField(null=True)
    last_referenced = models.DateTimeField(blank=True, null=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.original_file_name} {self.user}"

    def save(self, *args, **kwargs):
        if not self.last_referenced:
            if self.created_at:
                #  Needed to populate the initial last_referenced field for existing Files
                self.last_referenced = self.created_at
            else:
                self.last_referenced = timezone.now()
        super().save(*args, **kwargs)

    def delete(self, using=None, keep_parents=False):  # noqa: ARG002  # remove at Python 3.12
        #  Needed to make sure no orphaned files remain in the storage
        self.original_file.storage.delete(self.original_file.name)
        super().delete()

    def delete_from_s3(self):
        """Manually deletes the file from S3 storage."""
        self.original_file.delete(save=False)

    @property
    def file_type(self) -> str:
        name = self.original_file.name
        return name.split(".")[-1]

    @property
    def url(self) -> URL:
        #  In dev environment, get pre-signed url from minio
        if settings.ENVIRONMENT.uses_minio:
            s3 = boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
                config=Config(signature_version="s3v4"),
                region_name=settings.AWS_S3_REGION_NAME,
            )

            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": settings.AWS_STORAGE_BUCKET_NAME,
                    "Key": self.name,
                },
            )
            return URL(url)
        else:
            return URL(self.original_file.url)

    @property
    def name(self) -> str:
        # User-facing name
        return self.original_file_name or self.original_file.name

    @property
    def unique_name(self) -> str:
        # Name used by core-api
        return self.original_file.file.name

    def get_status_text(self) -> str:
        return next(
            (status[1] for status in StatusEnum.choices if self.status == status[0]),
            "Unknown",
        )

    @property
    def expiry_date(self) -> datetime:
        return self.last_referenced + timedelta(seconds=settings.FILE_EXPIRY_IN_SECONDS)


class ChatHistory(UUIDPrimaryKeyBase, TimeStampedModel):
    name = models.TextField(max_length=1024, null=False, blank=False)
    users = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name_plural = "Chat history"

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name} - {self.users}"


class ChatRoleEnum(models.TextChoices):
    ai = "ai"
    user = "user"
    system = "system"


class ChatMessage(UUIDPrimaryKeyBase, TimeStampedModel):
    chat_history = models.ForeignKey(ChatHistory, on_delete=models.CASCADE)
    text = models.TextField(max_length=32768, null=False, blank=False)
    role = models.CharField(choices=ChatRoleEnum.choices, null=False, blank=False)
    source_files = models.ManyToManyField(
        File,
        related_name="chat_messages",
        blank=True,
    )
    selected_files = models.ManyToManyField(File, related_name="+", symmetrical=False, blank=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.chat_history} - {self.text} - {self.role}"
