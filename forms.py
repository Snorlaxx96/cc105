from django import forms

# These choices would ideally be populated from the unique values
# extracted and saved by your notebook. For very long lists (like Publisher),
# you might show a subset or use an autocomplete widget in the template.
# Example: unique_platforms = ['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', ...]
# unique_genres = ['Sports', 'Platform', 'Racing', 'Role-Playing', ...]
# unique_publishers = ['Nintendo', 'Microsoft Game Studios', 'Take-Two Interactive', ...]

PLATFORM_CHOICES = [
    ('Wii', 'Wii'), ('NES', 'NES'), ('GB', 'GB'), ('DS', 'DS'),
    ('X360', 'Xbox 360'), ('PS3', 'PlayStation 3'), ('PS2', 'PlayStation 2'),
    # Add all platforms from your unique_platforms list [cite: 1]
]
GENRE_CHOICES = [
    ('Sports', 'Sports'), ('Platform', 'Platform'), ('Racing', 'Racing'),
    ('Role-Playing', 'Role-Playing'), ('Puzzle', 'Puzzle'), ('Misc', 'Misc'),
    # Add all genres from your unique_genres list [cite: 1]
]
PUBLISHER_CHOICES = [
    ('Nintendo', 'Nintendo'), ('Microsoft Game Studios', 'Microsoft Game Studios'),
    ('Take-Two Interactive', 'Take-Two Interactive'), ('Sony Computer Entertainment', 'Sony Computer Entertainment'),
    # Add publishers; consider a subset or autocomplete for long lists [cite: 1]
]

class GamePredictionForm(forms.Form):
    platform = forms.ChoiceField(choices=PLATFORM_CHOICES, label="Platform")
    genre = forms.ChoiceField(choices=GENRE_CHOICES, label="Genre")
    # For Publisher, due to the potentially large number of unique values,
    # you might want to use a CharField and handle it in the view,
    # or use a Django widget that supports autocomplete.
    # For simplicity here, a ChoiceField with a small subset.
    publisher = forms.ChoiceField(choices=PUBLISHER_CHOICES, label="Publisher")
    year = forms.IntegerField(label="Year of Release", min_value=1970, max_value=2025) # Adjust min/max as needed
    na_sales = forms.FloatField(label="North America Sales (in millions)", min_value=0)
    eu_sales = forms.FloatField(label="Europe Sales (in millions)", min_value=0)
    jp_sales = forms.FloatField(label="Japan Sales (in millions)", min_value=0)
    other_sales = forms.FloatField(label="Other Sales (in millions)", min_value=0)


    # game/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm # Import UserCreationForm

# ... (your GamePredictionForm is already here) ...

class SignUpForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')
    # You can add more fields here if you want (e.g., first_name, last_name)
    # and then include them in the Meta class fields list.

    class Meta(UserCreationForm.Meta):
        # model = User # User model is already handled by UserCreationForm
        fields = UserCreationForm.Meta.fields + ('email',) # Add email to the default fields (username, password1, password2)